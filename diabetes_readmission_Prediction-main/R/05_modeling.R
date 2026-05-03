library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(lightgbm)
library(smotefamily)
library(pROC)

set.seed(42)

df <- read_csv("data/processed/diabetes_featured.csv") %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(readmitted_binary = as.factor(ifelse(readmitted_binary == 1, "yes", "no")))

cat("=== MODELING ===\n")
cat("Dataset:", nrow(df), "rows,", ncol(df), "features\n")
cat("Class distribution:\n")
print(prop.table(table(df$readmitted_binary)))

# ── Stratified 70/30 split ────────────────────────────────────────────────────
train_idx <- createDataPartition(df$readmitted_binary, p = 0.7, list = FALSE)
train_raw <- df[train_idx, ]
test      <- df[-train_idx, ]
cat("Train:", nrow(train_raw), "| Test:", nrow(test), "\n")

# ── SMOTE on training set only ────────────────────────────────────────────────
cat("\nApplying SMOTE (K=5)...\n")
train_features <- train_raw %>% select(-readmitted_binary)
train_label    <- ifelse(train_raw$readmitted_binary == "yes", 1, 0)
train_num      <- train_features %>% mutate(across(where(is.factor), as.integer))

smote_result      <- SMOTE(train_num, train_label, K = 5, dup_size = 0)
train_smote       <- smote_result$data
train_smote_label <- as.factor(ifelse(train_smote$class == 1, "yes", "no"))
train_smote       <- train_smote %>% select(-class)

cat("After SMOTE — class distribution:\n")
print(prop.table(table(train_smote_label)))

# rebuild factor columns for LR and RF (caret needs factors)
train_smote_df <- bind_cols(train_smote, readmitted_binary = train_smote_label) %>%
  mutate(across(everything(), ~ if (is.numeric(.) && n_distinct(.) < 20) as.factor(.) else .)) %>%
  mutate(readmitted_binary = train_smote_label)
# Drop single-level factor columns that cause contrasts error
single_level <- sapply(train_smote_df, function(x) is.factor(x) && length(levels(droplevels(x))) < 2)
train_smote_df <- train_smote_df[, !single_level]

# ── 5-fold CV setup ──────────────────────────────────────────────────────────
ctrl <- trainControl(
  method = "cv", number = 5, classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final", verboseIter = FALSE
)
lr_grid <- expand.grid(
  alpha  = c(0, 0.5, 1),                            # ridge, elastic net, lasso
  lambda = c(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5)
)
model_lr <- train(
  readmitted_binary ~ ., data = train_smote_df,
  method = "glmnet", trControl = ctrl, metric = "ROC",
  tuneGrid = lr_grid, preProcess = c("center", "scale")
)
cat("LR best ROC:", round(max(model_lr$results$ROC), 3),
    "| alpha:", model_lr$bestTune$alpha,
    "| lambda:", model_lr$bestTune$lambda, "\n")

# ── [2/4] Random Forest ───────────────────────────────────────────────────────
cat("\n[2/4] Training Random Forest...\n")
rf_grid <- expand.grid(mtry = c(5, 8, 12, 15, 20))
model_rf <- train(
  readmitted_binary ~ ., data = train_smote_df,
  method = "rf", trControl = ctrl, metric = "ROC",
  tuneGrid = rf_grid, ntree = 500,          # 500 trees (was 200)
  classwt = c("no" = 1, "yes" = 3)         # additional class weighting on top of SMOTE
)
cat("RF best ROC:", round(max(model_rf$results$ROC), 3),
    "| mtry:", model_rf$bestTune$mtry, "\n")

# ── [3/4] XGBoost ────────────────────────────────────────────────────────────
cat("\n[3/4] Training XGBoost...\n")

xgb_features <- train_smote %>% mutate(across(everything(), as.numeric))
xgb_features <- xgb_features[, sapply(xgb_features, function(x) length(unique(x)) > 1)]
xgb_label    <- ifelse(train_smote_label == "yes", 1, 0)
xgb_mat      <- model.matrix(~ . - 1, data = xgb_features)

test_num <- test %>% select(-readmitted_binary) %>%
  mutate(across(where(is.factor), as.integer)) %>%
  mutate(across(everything(), as.numeric))
test_num <- test_num[, sapply(test_num, function(x) length(unique(x)) > 1)]
test_mat <- model.matrix(~ . - 1, data = test_num)
common_cols <- intersect(colnames(xgb_mat), colnames(test_mat))
xgb_mat  <- xgb_mat[, common_cols]
test_mat <- test_mat[, common_cols]
saveRDS(common_cols, "outputs/results/xgb_col_names.rds")
pos_weight <- sum(xgb_label == 0) / sum(xgb_label == 1)
cat("XGB scale_pos_weight:", round(pos_weight, 2), "\n")

dtrain <- xgb.DMatrix(data = xgb_mat, label = xgb_label)
dtest  <- xgb.DMatrix(data = test_mat,
                      label = ifelse(test$readmitted_binary == "yes", 1, 0))

# Expanded grid: adds min_child_weight and gamma for regularization
xgb_params_grid <- expand.grid(
  max_depth       = c(3, 5, 6, 8),
  eta             = c(0.01, 0.05, 0.1),
  min_child_weight = c(1, 5, 10),
  gamma           = c(0, 0.1, 0.5)
)
cat("XGB grid size:", nrow(xgb_params_grid), "combos\n")

best_auc <- 0; best_params <- NULL; best_nrounds <- 100
for (i in seq_len(nrow(xgb_params_grid))) {
  params <- list(
    objective         = "binary:logistic",
    eval_metric       = "auc",
    max_depth         = xgb_params_grid$max_depth[i],
    eta               = xgb_params_grid$eta[i],
    min_child_weight  = xgb_params_grid$min_child_weight[i],
    gamma             = xgb_params_grid$gamma[i],
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    lambda            = 1.0,       # L2 regularization
    alpha             = 0.1,       # L1 regularization
    scale_pos_weight  = pos_weight, # native class imbalance handling
    seed              = 42
  )
  cv_result <- xgb.cv(
    params = params, data = dtrain, nrounds = 300,
    nfold = 5, early_stopping_rounds = 30, verbose = FALSE
  )
  auc_val <- max(cv_result$evaluation_log$test_auc_mean)
  if (i %% 12 == 0 || auc_val > best_auc) {
    cat("  [", i, "/", nrow(xgb_params_grid), "]",
        "depth=", xgb_params_grid$max_depth[i],
        "eta=", xgb_params_grid$eta[i],
        "mcw=", xgb_params_grid$min_child_weight[i],
        "gamma=", xgb_params_grid$gamma[i],
        "-> AUC:", round(auc_val, 4), "\n")
  }
  if (auc_val > best_auc) {
    best_auc     <- auc_val
    best_params  <- params
    best_nrounds <- cv_result$best_iteration
  }
}

model_xgb <- xgb.train(
  params  = best_params, data = dtrain,
  nrounds = best_nrounds, verbose = 0
)
cat("XGB best CV AUC:", round(best_auc, 3), "| nrounds:", best_nrounds, "\n")
xgb.save(model_xgb, "outputs/results/model_xgb.json")

# ── [4/4] LightGBM ───────────────────────────────────────────────────────────
cat("\n[4/4] Training LightGBM...\n")
lgbm_train <- lgb.Dataset(data = xgb_mat, label = xgb_label)

# Expanded grid: adds min_child_samples and bagging options
lgbm_grid <- expand.grid(
  num_leaves       = c(31, 63, 127),
  learning_rate    = c(0.01, 0.05, 0.1),
  min_data_in_leaf = c(10, 20, 50)
)
cat("LightGBM grid size:", nrow(lgbm_grid), "combos\n")

best_lgbm_auc <- 0; best_lgbm_params <- NULL; best_lgbm_rounds <- 100
for (i in seq_len(nrow(lgbm_grid))) {
  lgbm_params <- list(
    objective         = "binary",
    metric            = "auc",
    num_leaves        = lgbm_grid$num_leaves[i],
    learning_rate     = lgbm_grid$learning_rate[i],
    min_data_in_leaf  = lgbm_grid$min_data_in_leaf[i],
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 0.1,    # L1 regularization
    lambda_l2         = 1.0,    # L2 regularization
    scale_pos_weight  = pos_weight,  # native class imbalance handling
    verbose           = -1
  )
  cv_result <- lgb.cv(
    params = lgbm_params, data = lgbm_train, nrounds = 300,
    nfold = 5, early_stopping_rounds = 30, verbose = -1
  )
  auc_val <- max(unlist(cv_result$record_evals$valid$auc$eval))
  if (i %% 9 == 0 || auc_val > best_lgbm_auc) {
    cat("  [", i, "/", nrow(lgbm_grid), "]",
        "leaves=", lgbm_grid$num_leaves[i],
        "lr=", lgbm_grid$learning_rate[i],
        "min_leaf=", lgbm_grid$min_data_in_leaf[i],
        "-> AUC:", round(auc_val, 4), "\n")
  }
  if (auc_val > best_lgbm_auc) {
    best_lgbm_auc    <- auc_val
    best_lgbm_params <- lgbm_params
    best_lgbm_rounds <- cv_result$best_iter
  }
}

model_lgbm <- lgb.train(
  params  = best_lgbm_params, data = lgbm_train,
  nrounds = best_lgbm_rounds, verbose = -1
)
cat("LightGBM best CV AUC:", round(best_lgbm_auc, 3), "\n")

# ── Save all artifacts ────────────────────────────────────────────────────────
saveRDS(model_lr,   "outputs/results/model_lr.rds")
saveRDS(model_rf,   "outputs/results/model_rf.rds")
saveRDS(model_xgb,  "outputs/results/model_xgb.rds")
lgb.save(model_lgbm, "outputs/results/model_lgbm.txt")
saveRDS(test,           "outputs/results/test_set.rds")
saveRDS(test_mat,       "outputs/results/test_mat.rds")
saveRDS(ifelse(test$readmitted_binary == "yes", 1, 0), "outputs/results/test_label.rds")
saveRDS(train_smote_df, "outputs/results/train_smote.rds")
saveRDS(xgb_mat,        "outputs/results/train_mat.rds")

cat("\nAll models saved to outputs/results/\n")
cat("Grid summary:\n")
cat("  LR:       18 combos (alpha x lambda)\n")
cat("  RF:        5 combos (mtry), ntree=500\n")
cat("  XGBoost: ", nrow(xgb_params_grid), "combos (max_depth x eta x min_child_weight x gamma)\n")
cat("  LightGBM:", nrow(lgbm_grid), "combos (num_leaves x learning_rate x min_data_in_leaf)\n")
