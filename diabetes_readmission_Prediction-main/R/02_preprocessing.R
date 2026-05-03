library(tidyverse)
library(janitor)

df <- read_csv("data/raw/diabetic_data.csv", na = c("", "NA", "?"),
               show_col_types = FALSE)
df <- df %>% clean_names()

cat("=== PREPROCESSING PIPELINE ===\n")
cat("Start:", nrow(df), "rows x", ncol(df), "cols\n")

# ── Step 1: Remove deceased / hospice patients ────────────────────────────────
# These patients cannot be readmitted — keeping them inflates the negative class
dead_ids <- c(11, 13, 14, 19, 20, 21)
df <- df %>% filter(!discharge_disposition_id %in% dead_ids)
cat("Step 1 — remove deceased/hospice:", nrow(df), "rows\n")

# ── Step 2: Deduplicate — keep chronologically FIRST encounter per patient ────
# Sort by encounter_id (sequential = chronological) so distinct() keeps the
# earliest visit. Using patient_nbr order alone is NOT chronological.
df <- df %>%
  arrange(patient_nbr, encounter_id) %>%
  distinct(patient_nbr, .keep_all = TRUE) %>%
  select(-patient_nbr, -encounter_id)
cat("Step 2 — dedup (first encounter per patient):", nrow(df), "rows\n")

# ── Step 3: Drop high-missing and zero-variance columns ───────────────────────
# weight (96.9% missing), payer_code (39.6% missing)
# examide / citoglipton / troglitazone / acetohexamide: 100% "No" → zero variance
# tolazamide / tolbutamide: >99.96% "No" → near-zero variance, pure noise
df <- df %>% select(
  -weight, -payer_code,
  -examide, -citoglipton, -troglitazone, -acetohexamide,
  -tolazamide, -tolbutamide
)
cat("Step 3 — dropped 8 useless cols (4 zero-var meds + 2 high-missing)\n")

# ── Step 4: Fix gender Unknown/Invalid ───────────────────────────────────────
# Only 3 patients — recode to majority class ("Female") to avoid a spurious level
df <- df %>%
  mutate(gender = ifelse(gender == "Unknown/Invalid", "Female", gender))

# ── Step 5: Impute remaining sparse missings ───────────────────────────────────
df <- df %>%
  mutate(
    race              = replace_na(race, "Unknown"),
    medical_specialty = replace_na(medical_specialty, "Unknown")
  )
cat("Step 5 — imputed race (2.2%) and medical_specialty (49.1%) → 'Unknown'\n")

# ── Step 6: Binary target ─────────────────────────────────────────────────────
df <- df %>%
  mutate(readmitted_binary = as.integer(readmitted == "<30")) %>%
  select(-readmitted)
cat("Step 6 — target: readmitted_binary\n")
cat("         positive rate:", round(mean(df$readmitted_binary) * 100, 1), "%\n")

# ── Step 7: Age range → numeric midpoint ──────────────────────────────────────
df <- df %>%
  mutate(age_numeric = case_when(
    age == "[0-10)"   ~  5, age == "[10-20)" ~ 15,
    age == "[20-30)"  ~ 25, age == "[30-40)" ~ 35,
    age == "[40-50)"  ~ 45, age == "[50-60)" ~ 55,
    age == "[60-70)"  ~ 65, age == "[70-80)" ~ 75,
    age == "[80-90)"  ~ 85, age == "[90-100)" ~ 95,
    TRUE ~ 65
  )) %>%
  select(-age)

# ── Step 8: ICD-9 diagnosis grouping ─────────────────────────────────────────
# NA diagnoses → "Missing" (distinct from the "Other" catch-all category)
group_diag <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  case_when(
    is.na(x)                              ~ "Missing",
    str_detect(x, "^[Vv]|^[Ee]")         ~ "Other",
    x_num >= 390 & x_num <= 459          ~ "Circulatory",
    x_num >= 460 & x_num <= 519          ~ "Respiratory",
    x_num >= 520 & x_num <= 579          ~ "Digestive",
    x_num >= 250 & x_num <  251          ~ "Diabetes",
    x_num >= 800 & x_num <= 999          ~ "Injury",
    x_num >= 710 & x_num <= 739          ~ "Musculoskeletal",
    x_num >= 580 & x_num <= 629          ~ "Genitourinary",
    x_num >= 140 & x_num <= 239          ~ "Neoplasms",
    TRUE                                  ~ "Other"
  )
}
df <- df %>% mutate(across(c(diag_1, diag_2, diag_3), group_diag))
cat("Step 8 — ICD-9 → 10 groups (+ 'Missing' for NA, was merged into 'Other')\n")

# ── Step 9: Medical specialty → clinical groups ───────────────────────────────
# 72 levels → 9 meaningful clinical categories (better than top-10 + Other)
specialty_group <- function(x) {
  case_when(
    str_detect(x, "InternalMedicine|Family|GeneralPractice") ~ "Internal_Family",
    str_detect(x, "Emergency|Trauma")                        ~ "Emergency",
    str_detect(x, "Cardiology|Cardiovascular")               ~ "Cardiology",
    str_detect(x, "Surgery")                                 ~ "Surgery",
    str_detect(x, "Nephrology|Urology")                      ~ "Renal_Urology",
    str_detect(x, "Orthopedics")                             ~ "Orthopedics",
    str_detect(x, "Pulmonology|Respiratory")                 ~ "Pulmonology",
    str_detect(x, "Psychiatry|Psychology")                   ~ "Psychiatry",
    x == "Unknown"                                           ~ "Unknown",
    TRUE                                                     ~ "Other"
  )
}
df <- df %>% mutate(medical_specialty = specialty_group(medical_specialty))
cat("Step 9 — medical specialty: 72 → 9 clinical groups\n")

# ── Step 10: A1C and glucose → ordinal ────────────────────────────────────────
df <- df %>%
  mutate(
    a1cresult     = case_when(
      a1cresult == "None" ~ 0L, a1cresult == "Norm" ~ 1L,
      a1cresult == ">7"   ~ 2L, a1cresult == ">8"   ~ 3L, TRUE ~ 0L),
    max_glu_serum = case_when(
      max_glu_serum == "None"  ~ 0L, max_glu_serum == "Norm"  ~ 1L,
      max_glu_serum == ">200"  ~ 2L, max_glu_serum == ">300"  ~ 3L, TRUE ~ 0L)
  )

# ── Step 11: Medication columns → ordinal (No=0, Steady=1, Up=2, Down=3) ──────
# Ordinal reflects prescribing intensity / dose adjustment activity.
# Tree models benefit from numeric encoding; LR gets meaningful coefficients.
med_cols <- c(
  "metformin", "repaglinide", "nateglinide", "chlorpropamide",
  "glimepiride", "glipizide", "glyburide", "pioglitazone",
  "rosiglitazone", "acarbose", "miglitol", "insulin",
  "glyburide_metformin", "glipizide_metformin",
  "glimepiride_pioglitazone", "metformin_rosiglitazone",
  "metformin_pioglitazone"
)
med_cols <- intersect(med_cols, names(df))

encode_med <- function(x) {
  case_when(
    x == "No"     ~ 0L,
    x == "Steady" ~ 1L,
    x == "Up"     ~ 2L,
    x == "Down"   ~ 3L,
    TRUE          ~ 0L
  )
}
df <- df %>% mutate(across(all_of(med_cols), encode_med))
cat("Step 11 — medications (", length(med_cols), "cols): factor → ordinal 0/1/2/3\n")

# ── Step 12: discharge_disposition_id → 7 clinical groups ────────────────────
# Raw 28 levels → clinically meaningful groups; preserves the signal of the
# strongest predictor while making it interpretable and reducing factor sparsity
df <- df %>%
  mutate(discharge_disposition_id = case_when(
    discharge_disposition_id == 1             ~ "Home",
    discharge_disposition_id %in% c(6, 8)    ~ "Home_Health",
    discharge_disposition_id %in% c(3, 15, 22, 24) ~ "SNF",
    discharge_disposition_id %in% c(2, 4, 5, 9, 10, 27) ~ "Transfer_Acute",
    discharge_disposition_id %in% c(16, 17, 23, 28, 29, 30) ~ "Transfer_Other",
    discharge_disposition_id == 7             ~ "AMA",
    TRUE                                      ~ "Other"
  ))
cat("Step 12 — discharge_disposition_id: 28 → 7 clinical groups\n")

# ── Step 13: admission_type_id → 4 clinical groups ───────────────────────────
df <- df %>%
  mutate(admission_type_id = case_when(
    admission_type_id == 1 ~ "Emergency",
    admission_type_id == 2 ~ "Urgent",
    admission_type_id == 3 ~ "Elective",
    TRUE                   ~ "Other"
  ))
cat("Step 13 — admission_type_id: 8 → 4 clinical groups\n")

# ── Step 14: admission_source_id → 5 clinical groups ─────────────────────────
df <- df %>%
  mutate(admission_source_id = case_when(
    admission_source_id == 7              ~ "ER",
    admission_source_id %in% c(1, 2, 3)  ~ "Physician_Referral",
    admission_source_id %in% c(4, 5, 6)  ~ "Transfer",
    admission_source_id == 17            ~ "HMO_Referral",
    TRUE                                  ~ "Other"
  ))
cat("Step 14 — admission_source_id: 17 → 5 clinical groups\n")

# ── Step 15: Numeric outlier capping (Winsorization at 99th percentile) ───────
# Extreme outliers (e.g., 76 emergency visits) distort LR standardization
# and can dominate tree splits. Cap at 99th pctile without removing rows.
winsorize <- function(x, p = 0.99) {
  cap <- quantile(x, p, na.rm = TRUE)
  pmin(x, cap)
}
df <- df %>%
  mutate(
    number_outpatient  = winsorize(number_outpatient),
    number_emergency   = winsorize(number_emergency),
    number_inpatient   = winsorize(number_inpatient),
    num_medications    = winsorize(num_medications),
    num_lab_procedures = winsorize(num_lab_procedures)
  )
cat("Step 15 — Winsorized 5 numeric cols at 99th percentile\n")

# ── Step 16: Convert remaining character / ID columns to factors ──────────────
df <- df %>%
  mutate(across(where(is.character), as.factor))
cat("Step 16 — character → factor\n")

# ── Final check ───────────────────────────────────────────────────────────────
cat("\nFinal dimensions:", nrow(df), "rows x", ncol(df), "cols\n")
cat("Missing values remaining:", sum(is.na(df)), "\n")
cat("Positive rate:", round(mean(df$readmitted_binary) * 100, 1), "%\n")
cat("\nColumn types:\n")
print(table(sapply(df, class)))

write_csv(df, "data/processed/diabetes_clean.csv")
cat("\nSaved: data/processed/diabetes_clean.csv\n")
