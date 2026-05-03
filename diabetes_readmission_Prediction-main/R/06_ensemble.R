library(tidyverse)
library(xgboost)
library(lightgbm)
library(pROC)
library(caret)
library(glmnet)

set.seed(42)

# load artifacts from modeling step
train_smote   <- readRDS("outputs/results/train_smote.rds")
train_mat     <- readRDS("outputs/results/train_mat.rds")
test_set      <- readRDS("outputs/results/test_set.rds")
model_lr      <- readRDS("outputs/results/model_lr.rds")
test_mat      <- readRDS("outputs/results/test_mat.rds")
test_label    <- readRDS("outputs/results/test_label.rds")
model_xgb     <- readRDS("outputs/results/model_xgb.rds")
model_lgbm    <- lgb.load("outputs/results/model_lgbm.txt")
model_rf      <- readRDS("outputs/results/model_rf.rds")
xgb_col_names <- readRDS("outputs/results/xgb_col_names.rds")

cat("=== STACKING ENSEMBLE ===\n")

# ── OOF predictions from RF (via caret savePredictions) ───────────────────────
oof_rf <- model_rf$pred %>%
  arrange(rowIndex) %>%
  group_by(rowIndex) %>%
  summarise(rf_prob = mean(yes), .groups = "drop") %>%
  pull(rf_prob)

# ── OOF for XGB and LightGBM: rerun 5-fold CV on train_mat ──────────────────
train_label_num <- ifelse(train_smote$readmitted_binary == "yes", 1, 0)
pos_weight      <- sum(train_label_num == 0) / sum(train_label_num == 1)

n        <- nrow(train_mat)
folds    <- createFolds(train_label_num, k = 5, list = TRUE)
oof_xgb  <- numeric(n)
oof_lgbm <- numeric(n)

xgb_params <- list(
  objective        = "binary:logistic", eval_metric = "auc",
  max_depth        = 6, eta = 0.05,
  min_child_weight = 5, gamma = 0.1,
  subsample        = 0.8, colsample_bytree = 0.8,
  lambda = 1.0, alpha = 0.1,
  scale_pos_weight = pos_weight, seed = 42
)
lgbm_params <- list(
  objective        = "binary", metric = "auc",
  num_leaves       = 63, learning_rate = 0.05,
  min_data_in_leaf = 20, feature_fraction = 0.8,
  bagging_fraction = 0.8, bagging_freq = 5,
  lambda_l1 = 0.1, lambda_l2 = 1.0,
  scale_pos_weight = pos_weight, verbose = -1
)

for (fold_i in seq_along(folds)) {
  val_idx   <- folds[[fold_i]]
  train_idx <- setdiff(seq_len(n), val_idx)
  
  d_tr  <- xgb.DMatrix(train_mat[train_idx, ], label = train_label_num[train_idx])
  d_val <- xgb.DMatrix(train_mat[val_idx,   ], label = train_label_num[val_idx])
  m_xgb <- xgb.train(params = xgb_params, data = d_tr, nrounds = 150, verbose = 0)
  oof_xgb[val_idx] <- predict(m_xgb, d_val)
  
  lgb_tr       <- lgb.Dataset(train_mat[train_idx, ], label = train_label_num[train_idx])
  m_lgbm       <- lgb.train(params = lgbm_params, data = lgb_tr, nrounds = 150, verbose = -1)
  oof_lgbm[val_idx] <- predict(m_lgbm, train_mat[val_idx, ])
  
  cat("Fold", fold_i, "done\n")
}

# ── Align OOF lengths ─────────────────────────────────────────────────────────
oof_len <- min(length(oof_rf), length(oof_xgb), length(oof_lgbm))
meta_train <- data.frame(
  rf   = oof_rf[1:oof_len],
  xgb  = oof_xgb[1:oof_len],
  lgbm = oof_lgbm[1:oof_len],
  y    = train_label_num[1:oof_len]
)

# ── Ridge meta-learner ────────────────────────────────────────────────────────
meta_x <- as.matrix(meta_train[, c("rf", "xgb", "lgbm")])
meta_y <- meta_train$y

cv_meta <- cv.glmnet(meta_x, meta_y, family = "binomial",
                     alpha = 0, nfolds = 5, type.measure = "auc")
cat("\nMeta-learner (Ridge) best lambda:", round(cv_meta$lambda.min, 5), "\n")
meta_model <- glmnet(meta_x, meta_y, family = "binomial",
                     alpha = 0, lambda = cv_meta$lambda.min)
cat("Meta-learner coefficients:\n")
print(coef(meta_model))

# ── Test set predictions from each base model ─────────────────────────────────
test_set_rf <- test_set %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(all_of(names(model_rf$xlevels)),
                ~ factor(as.character(.), levels = model_rf$xlevels[[cur_column()]]))) %>%
  mutate(across(c(race, gender, admission_type_id, discharge_disposition_id,
                  admission_source_id, medical_specialty, diag_1, diag_2,
                  diag_3, change, diabetes_med), as.integer))

rf_test_prob   <- predict(model_rf,  newdata = test_set_rf, type = "prob")[, "yes"]
xgb_test_prob  <- predict(model_xgb, xgb.DMatrix(test_mat))
lgbm_test_prob <- predict(model_lgbm, test_mat)

meta_test_x <- as.matrix(data.frame(
  rf   = rf_test_prob,
  xgb  = xgb_test_prob,
  lgbm = lgbm_test_prob
))
ensemble_prob <- as.numeric(
  predict(meta_model, newx = meta_test_x, type = "response")
)

cat("\nEnsemble test AUC:",
    round(auc(roc(test_label, ensemble_prob, quiet = TRUE)), 3), "\n")

# ── Threshold Optimization ────────────────────────────────────────────────────
cat("\n=== THRESHOLD OPTIMIZATION ===\n")

optimize_threshold <- function(probs, labels, model_name) {
  thresholds <- seq(0.01, 0.99, by = 0.01)
  metrics <- map_dfr(thresholds, function(t) {
    preds     <- as.integer(probs >= t)
    tp        <- sum(preds == 1 & labels == 1)
    fp        <- sum(preds == 1 & labels == 0)
    fn        <- sum(preds == 0 & labels == 1)
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall    <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1        <- ifelse(precision + recall == 0, 0,
                        2 * precision * recall / (precision + recall))
    tibble(threshold = t, precision = precision, recall = recall, f1 = f1)
  })
  
  f1_opt  <- metrics %>% slice_max(f1, n = 1) %>% pull(threshold)
  rec_opt <- metrics %>% filter(recall >= 0.70) %>%
    slice_max(f1, n = 1) %>% pull(threshold)
  rec_opt <- ifelse(length(rec_opt) == 0, f1_opt, rec_opt[1])
  
  cat(model_name, "-> F1-optimal:", f1_opt,
      "| recall>=0.70:", rec_opt, "\n")
  
  tibble(model              = model_name,
         threshold_default  = 0.5,
         threshold_f1_opt   = f1_opt,
         threshold_recall70 = rec_opt)
}

thresh_df <- bind_rows(
  optimize_threshold(xgb_test_prob,  test_label, "XGBoost"),
  optimize_threshold(lgbm_test_prob, test_label, "LightGBM"),
  optimize_threshold(ensemble_prob,  test_label, "Ensemble")
)
print(thresh_df)
write_csv(thresh_df, "outputs/results/optimal_thresholds.csv")

# ── Save ensemble artifacts ───────────────────────────────────────────────────
saveRDS(meta_model,    "outputs/results/model_ensemble.rds")
saveRDS(ensemble_prob, "outputs/results/ensemble_test_probs.rds")
saveRDS(meta_train,    "outputs/results/oof_predictions.rds")

cat("\nEnsemble and thresholds saved.\n")