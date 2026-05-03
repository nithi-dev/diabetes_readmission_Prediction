library(tidyverse)
library(skimr)
library(janitor)

set.seed(42)

# load raw data — ? is missing value code in this dataset
df <- read_csv("data/raw/diabetic_data.csv", na = c("", "NA", "?"))
df <- df %>% clean_names()

cat("=== RAW DATA DIMENSIONS ===\n")
cat("Rows:", nrow(df), " Cols:", ncol(df), "\n")

# full skim summary
skim_result <- skim(df)
print(skim_result)

# missing value audit
missing <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "col", values_to = "n_missing") %>%
  mutate(pct_missing = round(n_missing / nrow(df) * 100, 1)) %>%
  filter(n_missing > 0) %>%
  arrange(desc(pct_missing))

cat("\n=== MISSING VALUES ===\n")
print(missing, n = Inf)
write_csv(missing, "outputs/results/missing_audit.csv")

# missing heatmap (top 10 columns with missing data)
top_missing_cols <- missing %>% slice_head(n = 10) %>% pull(col)
if (length(top_missing_cols) > 0) {
  missing_mat <- df %>%
    select(all_of(top_missing_cols)) %>%
    slice_head(n = 2000) %>%
    mutate(row_id = row_number()) %>%
    pivot_longer(-row_id, names_to = "col", values_to = "val") %>%
    mutate(is_missing = is.na(val))

  ggplot(missing_mat, aes(x = col, y = row_id, fill = is_missing)) +
    geom_tile() +
    scale_fill_manual(values = c("FALSE" = "#CCCCCC", "TRUE" = "#D85A30"),
                      labels = c("present", "missing")) +
    labs(title = "missing data heatmap (sample of 2000 rows)",
         x = "", y = "row index", fill = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave("outputs/figures/00_missing_heatmap.png", width = 9, height = 5)
  cat("saved 00_missing_heatmap.png\n")
}

# target variable distribution
cat("\n=== TARGET VARIABLE (readmitted) ===\n")
df %>%
  count(readmitted) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

# deceased/hospice patients — will be removed in preprocessing
dead_ids <- c(11, 13, 14, 19, 20, 21)
dead_count <- df %>% filter(discharge_disposition_id %in% dead_ids) %>% nrow()
cat("\nDeceased/hospice patients (discharge IDs 11,13,14,19,20,21):", dead_count, "\n")
cat("These will be removed in preprocessing (cannot be readmitted)\n")

# duplicate patients
n_repeated <- df %>% count(patient_nbr) %>% filter(n > 1) %>% nrow()
cat("\n=== DUPLICATE PATIENTS ===\n")
cat("Unique patients:", n_distinct(df$patient_nbr), "\n")
cat("Total encounters:", nrow(df), "\n")
cat("Patients with >1 encounter:", n_repeated, "\n")

# cardinality of categorical columns
cat("\n=== CARDINALITY (character/factor columns) ===\n")
card <- df %>%
  select(where(is.character)) %>%
  summarise(across(everything(), n_distinct)) %>%
  pivot_longer(everything(), names_to = "col", values_to = "unique_vals") %>%
  arrange(desc(unique_vals))
print(card, n = Inf)
write_csv(card, "outputs/results/cardinality.csv")

# numeric summary
cat("\n=== NUMERIC SUMMARY ===\n")
df %>% select(where(is.numeric)) %>% summary() %>% print()

# save compact summary
data_summary <- tibble(
  metric = c("total_rows", "unique_patients", "repeated_patients",
             "deceased_hospice", "pct_readmit_lt30", "pct_readmit_gt30", "pct_not_readmit"),
  value  = c(
    nrow(df),
    n_distinct(df$patient_nbr),
    n_repeated,
    dead_count,
    round(mean(df$readmitted == "<30", na.rm = TRUE) * 100, 1),
    round(mean(df$readmitted == ">30", na.rm = TRUE) * 100, 1),
    round(mean(df$readmitted == "NO",  na.rm = TRUE) * 100, 1)
  ),
  unit = c("count", "count", "count", "count", "pct", "pct", "pct")
)
write_csv(data_summary, "outputs/results/data_summary.csv")
cat("\ndata_summary.csv saved\n")
