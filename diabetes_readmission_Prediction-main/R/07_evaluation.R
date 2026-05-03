library(tidyverse)
library(caret)
library(pROC)
library(PRROC)
library(xgboost)
library(lightgbm)
library(ggplot2)

model_lr   <- readRDS("outputs/results/model_lr.rds")
model_rf   <- readRDS("outputs/results/model_rf.rds")
model_xgb  <- readRDS("outputs/results/model_xgb.rds")
model_lgbm <- lgb.load("outputs/results/model_lgbm.txt")
model_ens  <- readRDS("outputs/results/model_ensemble.rds")
test_set   <- readRDS("outputs/results/test_set.rds")
test_mat   <- readRDS("outputs/results/test_mat.rds")
test_label <- readRDS("outputs/results/test_label.rds")
ens_probs  <- readRDS("outputs/results/ensemble_test_probs.rds")
thresh_df  <- read_csv("outputs/results/optimal_thresholds.csv",
                       show_col_types = FALSE)

COLORS <- c(LR = "#7F77DD", RF = "#1D9E75", XGB = "#D85A30",
            LightGBM = "#F4A261", Ensemble = "#6C3483")

# ── Predictions ───────────────────────────────────────────────────────────────
# Fix test_set types for LR prediction
test_set_lr <- test_set %>% mutate(across(c(race, gender, admission_type_id,
                                            discharge_disposition_id, admission_source_id, medical_specialty,
                                            diag_1, diag_2, diag_3, change, diabetes_med), as.integer))
for (col in names(model_lr$xlevels)) {
  if (col %in% names(test_set_lr)) {
    test_set_lr[[col]] <- factor(as.character(test_set_lr[[col]]),
                                 levels = model_lr$xlevels[[col]])
  }
}
lr_prob   <- predict(model_lr,   newdata = test_set_lr, type = "prob")[, "yes"]

# Fix test_set types for RF prediction
test_set_rf <- test_set %>% mutate(across(c(race, gender, admission_type_id,
                                            discharge_disposition_id, admission_source_id, medical_specialty,
                                            diag_1, diag_2, diag_3, change, diabetes_med), as.integer))
rf_prob   <- predict(model_rf,   newdata = test_set_rf, type = "prob")[, "yes"]
xgb_prob  <- predict(model_xgb,  xgb.DMatrix(test_mat))
lgbm_prob <- predict(model_lgbm, test_mat)

# ── Metrics function ──────────────────────────────────────────────────────────
eval_model <- function(probs, label, model_name, opt_thresh = NULL) {
  thresh <- if (!is.null(opt_thresh)) opt_thresh else 0.5
  preds  <- as.integer(probs >= thresh)
  tp  <- sum(preds == 1 & label == 1)
  fp  <- sum(preds == 1 & label == 0)
  fn  <- sum(preds == 0 & label == 1)
  tn  <- sum(preds == 0 & label == 0)
  precision   <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall      <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, 0, tn / (tn + fp))
  npv         <- ifelse(tn + fn == 0, 0, tn / (tn + fn))
  f1          <- ifelse(precision + recall == 0, 0,
                        2 * precision * recall / (precision + recall))
  roc_obj <- roc(label, probs, quiet = TRUE)
  pr_obj  <- pr.curve(scores.class0 = probs[label == 1],
                      scores.class1 = probs[label == 0], curve = FALSE)
  tibble(
    model       = model_name,
    threshold   = round(thresh, 3),
    auc         = round(auc(roc_obj), 3),
    pr_auc      = round(pr_obj$auc.integral, 3),
    precision   = round(precision, 3),
    recall      = round(recall, 3),
    specificity = round(specificity, 3),
    npv         = round(npv, 3),
    f1          = round(f1, 3)
  )
}

get_thresh <- function(model_name) {
  t <- thresh_df %>% filter(model == model_name) %>% pull(threshold_f1_opt)
  if (length(t) == 0) 0.5 else t[1]
}

results <- bind_rows(
  eval_model(lr_prob,   test_label, "LR",       0.5),
  eval_model(rf_prob,   test_label, "RF",       0.5),
  eval_model(xgb_prob,  test_label, "XGB",      get_thresh("XGBoost")),
  eval_model(lgbm_prob, test_label, "LightGBM", get_thresh("LightGBM")),
  eval_model(ens_probs, test_label, "Ensemble", get_thresh("Ensemble"))
)

cat("=== MODEL COMPARISON ===\n")
print(results, n = 10)
write_csv(results, "outputs/results/model_comparison.csv")

# ── Figure 17: ROC curves ─────────────────────────────────────────────────────
roc_lr   <- roc(test_label, lr_prob,   quiet = TRUE)
roc_rf   <- roc(test_label, rf_prob,   quiet = TRUE)
roc_xgb  <- roc(test_label, xgb_prob,  quiet = TRUE)
roc_lgbm <- roc(test_label, lgbm_prob, quiet = TRUE)
roc_ens  <- roc(test_label, ens_probs, quiet = TRUE)

png("outputs/figures/17_roc_curves.png", width = 800, height = 650)
plot(roc_lr,   col = COLORS["LR"],       lwd = 2,
     main = "ROC curves — all models",
     xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(roc_rf,   col = COLORS["RF"],       lwd = 2, add = TRUE)
plot(roc_xgb,  col = COLORS["XGB"],      lwd = 2, add = TRUE)
plot(roc_lgbm, col = COLORS["LightGBM"], lwd = 2, add = TRUE)
plot(roc_ens,  col = COLORS["Ensemble"], lwd = 2, add = TRUE, lty = 2)
abline(a = 0, b = 1, lty = 3, col = "grey60")
legend("bottomright", bty = "n",
  legend = c(
    paste("LR        AUC =", round(auc(roc_lr),   3)),
    paste("RF        AUC =", round(auc(roc_rf),   3)),
    paste("XGBoost  AUC =", round(auc(roc_xgb),  3)),
    paste("LightGBM AUC =", round(auc(roc_lgbm), 3)),
    paste("Ensemble AUC =", round(auc(roc_ens),  3))
  ),
  col = unname(COLORS), lwd = 2, cex = 0.9)
dev.off()
cat("saved 17_roc_curves.png\n")

# ── Figure 18: PR curves ──────────────────────────────────────────────────────
png("outputs/figures/18_pr_curves.png", width = 800, height = 650)
make_pr <- function(probs, label) {
  pr.curve(scores.class0 = probs[label == 1],
           scores.class1 = probs[label == 0], curve = TRUE)
}
pr_lr   <- make_pr(lr_prob,   test_label)
pr_rf   <- make_pr(rf_prob,   test_label)
pr_xgb  <- make_pr(xgb_prob,  test_label)
pr_lgbm <- make_pr(lgbm_prob, test_label)
pr_ens  <- make_pr(ens_probs, test_label)

plot(pr_lr,   col = COLORS["LR"],       lwd = 2, auc.main = FALSE,
     main = "Precision-Recall curves — all models")
plot(pr_rf,   col = COLORS["RF"],       lwd = 2, add = TRUE)
plot(pr_xgb,  col = COLORS["XGB"],      lwd = 2, add = TRUE)
plot(pr_lgbm, col = COLORS["LightGBM"], lwd = 2, add = TRUE)
plot(pr_ens,  col = COLORS["Ensemble"], lwd = 2, add = TRUE, lty = 2)
abline(h = mean(test_label), lty = 3, col = "grey60")
legend("topright", bty = "n",
  legend = c(
    paste("LR        PR-AUC =", round(pr_lr$auc.integral,   3)),
    paste("RF        PR-AUC =", round(pr_rf$auc.integral,   3)),
    paste("XGBoost  PR-AUC =", round(pr_xgb$auc.integral,  3)),
    paste("LightGBM PR-AUC =", round(pr_lgbm$auc.integral, 3)),
    paste("Ensemble PR-AUC =", round(pr_ens$auc.integral,  3))
  ),
  col = unname(COLORS), lwd = 2, cex = 0.9)
dev.off()
cat("saved 18_pr_curves.png\n")

# ── Figure 19: RF feature importance ─────────────────────────────────────────
rf_imp <- varImp(model_rf)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall)) %>%
  head(15)

ggplot(rf_imp, aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col(fill = COLORS["RF"], alpha = 0.85) +
  coord_flip() +
  labs(title = "top 15 features — Random Forest importance",
       x = "", y = "importance") +
  theme_minimal()
ggsave("outputs/figures/19_rf_feature_importance.png", width = 7, height = 6)
cat("saved 19_rf_feature_importance.png\n")

# ── Figure 20: LightGBM feature importance ───────────────────────────────────
lgbm_imp <- lgb.importance(model_lgbm) %>% head(15)

ggplot(lgbm_imp, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = COLORS["LightGBM"], alpha = 0.85) +
  coord_flip() +
  labs(title = "top 15 features — LightGBM (gain)",
       x = "", y = "gain") +
  theme_minimal()
ggsave("outputs/figures/20_lgbm_feature_importance.png", width = 7, height = 6)
cat("saved 20_lgbm_feature_importance.png\n")

# ── Figure 21: Threshold curve — Ensemble ────────────────────────────────────
thresh_curve <- map_dfr(seq(0.01, 0.99, by = 0.01), function(t) {
  preds <- as.integer(ens_probs >= t)
  tp <- sum(preds == 1 & test_label == 1)
  fp <- sum(preds == 1 & test_label == 0)
  fn <- sum(preds == 0 & test_label == 1)
  pr <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  re <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1 <- ifelse(pr + re == 0, 0, 2 * pr * re / (pr + re))
  tibble(threshold = t, precision = pr, recall = re, f1 = f1)
})

thresh_curve %>%
  pivot_longer(c(precision, recall, f1), names_to = "metric") %>%
  ggplot(aes(x = threshold, y = value, color = metric)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c(precision = "#D85A30", recall = "#1D9E75", f1 = "#7F77DD")) +
  geom_vline(xintercept = get_thresh("Ensemble"), linetype = "dashed", alpha = 0.7) +
  labs(title = "threshold vs metrics — Ensemble model",
       subtitle = "dashed = F1-optimal threshold",
       x = "threshold", y = "value", color = "metric") +
  theme_minimal()
ggsave("outputs/figures/21_threshold_curve.png", width = 7, height = 4)
cat("saved 21_threshold_curve.png\n")

# ── Figure 25: Calibration curves ────────────────────────────────────────────
# Calibration = do predicted probabilities match actual positive rates?
calibration_curve <- function(probs, labels, n_bins = 10) {
  bins <- cut(probs, breaks = seq(0, 1, length.out = n_bins + 1),
              include.lowest = TRUE)
  tibble(prob = probs, label = labels, bin = bins) %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(prob), mean_actual = mean(label),
              n = n(), .groups = "drop") %>%
    filter(n >= 10)  # drop bins with < 10 observations
}

cal_data <- bind_rows(
  calibration_curve(lr_prob,   test_label) %>% mutate(model = "LR"),
  calibration_curve(rf_prob,   test_label) %>% mutate(model = "RF"),
  calibration_curve(xgb_prob,  test_label) %>% mutate(model = "XGB"),
  calibration_curve(lgbm_prob, test_label) %>% mutate(model = "LightGBM"),
  calibration_curve(ens_probs, test_label) %>% mutate(model = "Ensemble")
) %>%
  mutate(model = factor(model, levels = c("LR", "RF", "XGB", "LightGBM", "Ensemble")))

ggplot(cal_data, aes(x = mean_pred, y = mean_actual, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_color_manual(values = unname(COLORS)) +
  labs(title = "calibration curves — predicted vs actual probability",
       subtitle = "dashed = perfect calibration",
       x = "mean predicted probability", y = "actual positive rate",
       color = "model") +
  theme_minimal() +
  coord_equal(xlim = c(0, 0.6), ylim = c(0, 0.6))
ggsave("outputs/figures/25_calibration_curves.png", width = 7, height = 6)
cat("saved 25_calibration_curves.png\n")

# ── DeLong test: ensemble vs each model ───────────────────────────────────────
cat("\n=== DeLong AUC Tests (vs Ensemble) ===\n")
for (nm in c("LR", "RF", "XGB", "LightGBM")) {
  other_roc <- switch(nm, LR = roc_lr, RF = roc_rf,
                          XGB = roc_xgb, LightGBM = roc_lgbm)
  test_res  <- roc.test(roc_ens, other_roc, method = "delong")
  cat(sprintf("Ensemble vs %s: p=%.4f %s\n",
              nm, test_res$p.value,
              ifelse(test_res$p.value < 0.05, "(significant)", "")))
}

cat("\nEvaluation complete. All results saved.\n")
