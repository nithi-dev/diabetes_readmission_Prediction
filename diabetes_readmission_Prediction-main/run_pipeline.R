# run_pipeline.R — execute the full diabetes readmission pipeline in one shot.
# Run from project root: source("run_pipeline.R")
# Total runtime: ~45-60 min (dominated by expanded grid-search in 05_modeling.R)
#   XGB: 144 combos | LightGBM: 27 combos | RF: 5 combos (ntree=500)

cat("========================================\n")
cat(" DIABETES READMISSION PREDICTION PIPELINE\n")
cat("========================================\n\n")

pipeline <- list(
  list(script = "R/01_data_audit.R",         label = "1/7  Data audit"),
  list(script = "R/02_preprocessing.R",      label = "2/7  Preprocessing"),
  list(script = "R/03_eda.R",                label = "3/7  EDA (16 plots)"),
  list(script = "R/04_feature_engineering.R",label = "4/7  Feature engineering"),
  list(script = "R/05_modeling.R",           label = "5/7  Modeling (grid search) ~45-60 min"),
  list(script = "R/06_ensemble.R",           label = "6/7  Ensemble + threshold opt."),
  list(script = "R/07_evaluation.R",         label = "7/7  Evaluation + figures")
)

for (step in pipeline) {
  cat(sprintf("[START] %s\n", step$label))
  t0 <- proc.time()
  source(step$script, local = FALSE)
  elapsed <- round((proc.time() - t0)[["elapsed"]])
  cat(sprintf("[DONE ] %s — %ds\n\n", step$label, elapsed))
}

# Export xgb column names as CSV so Python SHAP can align features
if (file.exists("outputs/results/xgb_col_names.rds")) {
  col_names <- readRDS("outputs/results/xgb_col_names.rds")
  write.csv(
    data.frame(col_names = col_names),
    "outputs/results/xgb_col_names.csv",
    row.names = FALSE
  )
  cat("Exported xgb_col_names.csv for Python SHAP\n")
}

cat("\n========================================\n")
cat(" R pipeline complete.\n")
cat(" Next: run Python SHAP analysis\n")
cat("   .venv/Scripts/python.exe python/shap_analysis.py\n")
cat(" Then: knit the report\n")
cat("   rmarkdown::render('docs/report.Rmd', output_file='report.html')\n")
cat("========================================\n")
