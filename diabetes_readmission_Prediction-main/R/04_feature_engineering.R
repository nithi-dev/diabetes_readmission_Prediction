library(tidyverse)

df <- read_csv("data/processed/diabetes_clean.csv")

cat("=== FEATURE ENGINEERING ===\n")
cat("Input:", nrow(df), "rows x", ncol(df), "cols\n")

# ── Medication columns (23 drug features) ─────────────────────────────────────
med_cols <- c(
  "metformin", "repaglinide", "nateglinide", "chlorpropamide",
  "glimepiride", "acetohexamide", "glipizide", "glyburide",
  "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
  "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
  "insulin", "glyburide_metformin", "glipizide_metformin",
  "glimepiride_pioglitazone", "metformin_rosiglitazone",
  "metformin_pioglitazone"
)
med_cols <- intersect(med_cols, names(df))

df <- df %>%
  mutate(
    # ── Original features ───────────────────────────────────────────────────

    # 1. High utilizer — strong prior signal for readmission
    high_utilizer    = as.integer(number_inpatient >= 3),

    # 2. Polypharmacy flag — medication complexity marker
    polypharmacy     = as.integer(num_medications > 10),

    # 3. Total healthcare visits — combined utilization score
    total_visits     = number_outpatient + number_emergency + number_inpatient,

    # 4. Diabetes as primary diagnosis
    diab_primary     = as.integer(diag_1 == "Diabetes"),

    # 5. Medication changed during encounter
    any_change       = as.integer(change == "Ch"),

    # 6. Age × inpatient interaction (elderly + frequent admissions = high risk)
    age_x_inpatient  = age_numeric * number_inpatient,

    # 7. Medications × diagnoses (complexity composite)
    med_x_diagnoses  = num_medications * number_diagnoses,

    # 8. Emergency ratio — divide-by-zero guarded
    emergency_ratio  = ifelse(
      number_outpatient + number_emergency + number_inpatient > 0,
      number_emergency / (number_outpatient + number_emergency + number_inpatient),
      0
    ),

    # ── New features ────────────────────────────────────────────────────────

    # 9. Charlson Comorbidity Index proxy from ICD-9 groups
    #    Higher score → more severe comorbidity → higher readmission risk
    charlson_proxy = (
      as.integer(diag_1 == "Circulatory"     | diag_2 == "Circulatory"     | diag_3 == "Circulatory")     * 1 +
      as.integer(diag_1 == "Respiratory"     | diag_2 == "Respiratory"     | diag_3 == "Respiratory")     * 1 +
      as.integer(diag_1 == "Diabetes"        | diag_2 == "Diabetes"        | diag_3 == "Diabetes")        * 1 +
      as.integer(diag_1 == "Neoplasms"       | diag_2 == "Neoplasms"       | diag_3 == "Neoplasms")       * 2 +
      as.integer(diag_1 == "Genitourinary"   | diag_2 == "Genitourinary"   | diag_3 == "Genitourinary")   * 1 +
      as.integer(diag_1 == "Musculoskeletal" | diag_2 == "Musculoskeletal" | diag_3 == "Musculoskeletal") * 1
    ),

    # 10. Multi-system disease — comorbidity span across diag_1/2/3
    n_diag_systems   = as.integer(diag_1 != "Other") +
                       as.integer(diag_2 != "Other") +
                       as.integer(diag_3 != "Other"),

    # 11. Prior visit intensity — weighted sum (inpatient > emergency > outpatient)
    visit_intensity  = number_inpatient * 3 + number_emergency * 2 + number_outpatient * 1,

    # 12. A1C tested AND elevated — poor glucose control at admission
    poor_glucose_ctrl = as.integer(a1cresult >= 2),   # a1cresult: 0=None,1=Norm,2=>7,3=>8

    # 13. High medication burden AND high diagnosis count (compound complexity)
    complex_case     = as.integer(num_medications > 10 & number_diagnoses >= 7),

    # 14. Insulin-dependent diabetic (insulin prescribed or steady)
    insulin_dependent = as.integer(insulin %in% c("Up", "Down", "Steady"))
  )

# 15. Medication change diversity — count of distinct meds with any change
#     (Up/Down = actively adjusted; correlates with instability)
if (length(med_cols) > 0) {
  med_df <- df %>% select(all_of(med_cols))
  df$n_meds_adjusted <- rowSums(
    sapply(med_df, function(x) as.integer(x %in% c("Up", "Down")))
  )
} else {
  df$n_meds_adjusted <- 0L
}

cat("New features (15 total):\n")
cat("  Original 8: high_utilizer, polypharmacy, total_visits, diab_primary,\n")
cat("              any_change, age_x_inpatient, med_x_diagnoses, emergency_ratio\n")
cat("  New 7: charlson_proxy, n_diag_systems, visit_intensity,\n")
cat("         poor_glucose_ctrl, complex_case, insulin_dependent, n_meds_adjusted\n")
cat("Output:", nrow(df), "rows x", ncol(df), "cols\n")

write_csv(df, "data/processed/diabetes_featured.csv")
cat("Saved: data/processed/diabetes_featured.csv\n")
