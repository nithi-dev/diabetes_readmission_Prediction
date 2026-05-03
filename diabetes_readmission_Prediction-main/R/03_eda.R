library(tidyverse)
library(scales)
library(ggcorrplot)

df <- read_csv("data/processed/diabetes_clean.csv") %>% 
  mutate(readmitted_binary = as.factor(as.integer(readmitted_binary)))

COLORS <- c("0" = "#7F77DD", "1" = "#D85A30")
theme_set(theme_minimal(base_size = 12))

save_fig <- function(name, w = 7, h = 5) {
  ggsave(paste0("outputs/figures/", name), width = w, height = h, dpi = 150)
  cat("saved:", name, "\n")
}

# 01: Class distribution
df %>% count(readmitted_binary) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  ggplot(aes(x = readmitted_binary, y = n, fill = readmitted_binary)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(pct, "%")), vjust = -0.4, size = 4.5) +
  scale_fill_manual(values = COLORS) +
  labs(title = "class distribution — readmission within 30 days",
       x = "readmitted <30 days", y = "count") +
  theme(legend.position = "none")
save_fig("01_class_distribution.png", 6, 4)

# 02: readmission rate by age group
df %>%
  mutate(age_group = cut(age_numeric,
    breaks = c(0,20,40,60,80,100),
    labels = c("<20","20-40","40-60","60-80","80+"))) %>%
  group_by(age_group) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = age_group, y = readmit_rate)) +
  geom_col(fill = "#D85A30", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            vjust = -0.4, size = 3.5) +
  scale_y_continuous(labels = percent) +
  labs(title = "readmission rate by age group",
       x = "age group", y = "readmission rate")
save_fig("02_age_readmit_rate.png", 7, 4)

# 03: time in hospital by class
ggplot(df, aes(x = readmitted_binary, y = time_in_hospital, fill = readmitted_binary)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = COLORS) +
  labs(title = "time in hospital vs readmission",
       x = "readmitted <30 days", y = "days") +
  theme(legend.position = "none")
save_fig("03_time_in_hospital.png", 6, 4)

# 04: prior inpatient visits by class
ggplot(df, aes(x = readmitted_binary, y = number_inpatient, fill = readmitted_binary)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = COLORS) +
  labs(title = "prior inpatient visits vs readmission",
       x = "readmitted <30 days", y = "prior inpatient visits") +
  theme(legend.position = "none")
save_fig("04_prior_inpatient.png", 6, 4)

# 05: number of medications density
ggplot(df, aes(x = num_medications, fill = readmitted_binary)) +
  geom_density(alpha = 0.55) +
  scale_fill_manual(values = COLORS, labels = c("not readmitted","readmitted <30")) +
  labs(title = "medication count by readmission status",
       x = "number of medications", y = "density", fill = "")
save_fig("05_num_medications.png", 7, 4)

# 06: readmission rate by primary diagnosis
df %>%
  group_by(diag_1) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  filter(n > 200) %>%
  ggplot(aes(x = reorder(diag_1, readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#D85A30", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 0.15)) +
  labs(title = "readmission rate by primary diagnosis (ICD-9 group)",
       x = "", y = "readmission rate")
save_fig("06_readmit_by_diag.png", 7, 5)

# 07: insulin use vs readmission
df %>%
  group_by(insulin) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = reorder(as.character(insulin), readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#1D9E75", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 0.12)) +
  labs(title = "readmission rate by insulin prescription",
       x = "insulin", y = "readmission rate")
save_fig("07_insulin_readmit.png", 6, 4)

# 08: A1C result vs readmission
df %>%
  mutate(a1c_label = case_when(
    a1cresult == 0 ~ "Not tested",
    a1cresult == 1 ~ "Normal",
    a1cresult == 2 ~ ">7",
    a1cresult == 3 ~ ">8"
  )) %>%
  group_by(a1c_label) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = reorder(a1c_label, readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#5B9BD5", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 0.12)) +
  labs(title = "readmission rate by A1C result",
       x = "A1C result", y = "readmission rate")
save_fig("08_a1c_readmit.png", 6, 4)

# 09: medication change vs readmission
df %>%
  group_by(change) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = as.character(change), y = readmit_rate,
             fill = as.character(change))) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(values = c("Ch" = "#D85A30", "No" = "#7F77DD")) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            vjust = -0.4, size = 4) +
  scale_y_continuous(labels = percent, limits = c(0, 0.12)) +
  labs(title = "readmission rate: medication change during visit",
       x = "medication change (Ch = changed)", y = "readmission rate") +
  theme(legend.position = "none")
save_fig("09_med_change_readmit.png", 6, 4)

# 10: discharge disposition clinical group vs readmission rate
df %>%
  group_by(discharge_disposition_id) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = reorder(as.character(discharge_disposition_id), readmit_rate),
             y = readmit_rate)) +
  geom_col(fill = "#F4A261", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 0.25)) +
  labs(title = "readmission rate by discharge disposition group",
       x = "discharge group", y = "readmission rate")
save_fig("10_discharge_readmit.png", 7, 5)

num_df <- select(df, where(is.numeric))
num_df$readmitted <- as.integer(as.character(df$readmitted_binary))




cor_mat <- cor(num_df, use = "complete.obs")
ggcorrplot(cor_mat, method = "square", type = "lower",
           lab = TRUE, lab_size = 2.5,
           colors = c("#7F77DD", "white", "#D85A30"),
           title = "correlation matrix — numeric features")
save_fig("11_correlation_heatmap.png", 9, 8)

# 12: number of diagnoses
ggplot(df, aes(x = number_diagnoses, fill = readmitted_binary)) +
  geom_bar(position = "fill", alpha = 0.85) +
  scale_fill_manual(values = COLORS, labels = c("not readmitted","readmitted <30")) +
  scale_y_continuous(labels = percent) +
  labs(title = "readmission proportion by number of diagnoses",
       x = "number of diagnoses", y = "proportion", fill = "")
save_fig("12_num_diagnoses.png", 7, 4)

# 13: race vs readmission
df %>%
  group_by(race) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  filter(n > 100) %>%
  ggplot(aes(x = reorder(as.character(race), readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#7F77DD", alpha = 0.85) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 0.12)) +
  labs(title = "readmission rate by race",
       x = "", y = "readmission rate")
save_fig("13_race_readmit.png", 7, 4)

# 14: number of emergency visits
ggplot(df, aes(x = readmitted_binary, y = number_emergency, fill = readmitted_binary)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2) +
  scale_fill_manual(values = COLORS) +
  labs(title = "emergency visits vs readmission",
       x = "readmitted <30 days", y = "number of emergency visits") +
  theme(legend.position = "none")
save_fig("14_emergency_visits.png", 6, 4)

# 15: polypharmacy (>10 medications) vs readmission
df %>%
  mutate(polypharmacy = ifelse(num_medications > 10, ">10 meds", "<=10 meds")) %>%
  group_by(polypharmacy) %>%
  summarise(readmit_rate = mean(as.integer(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = polypharmacy, y = readmit_rate, fill = polypharmacy)) +
  geom_col(alpha = 0.85, width = 0.5) +
  geom_text(aes(label = scales::percent(readmit_rate, accuracy = 0.1)),
            vjust = -0.4, size = 4.5) +
  scale_y_continuous(labels = percent, limits = c(0, 0.12)) +
  scale_fill_manual(values = c("<=10 meds" = "#7F77DD", ">10 meds" = "#D85A30")) +
  labs(title = "polypharmacy vs readmission rate",
       x = "", y = "readmission rate") +
  theme(legend.position = "none")
save_fig("15_polypharmacy.png", 6, 4)

# --- Statistical Tests ---
cat("\n=== STATISTICAL TESTS ===\n")

# chi-square for categorical columns
cat_cols <- df %>% select(where(is.character)) %>% names()


chi_results <- map_dfr(cat_cols, function(col) {
  tbl <- table(df[[col]], df$readmitted_binary)
  res <- stats::chisq.test(tbl, simulate.p.value = TRUE)
  tibble(feature = col, test = "chi-square",
         statistic = round(as.numeric(res$statistic), 2), p_value = round(res$p.value, 4))
})

# Mann-Whitney for numeric columns
num_cols <- setdiff(names(select(df, where(is.numeric))), "readmitted_binary")


mw_results <- map_dfr(num_cols, function(col) {
  wt <- wilcox.test(df[[col]] ~ df$readmitted_binary, exact = FALSE)
  tibble(feature = col, test = "mann-whitney",
         statistic = round(as.numeric(wt$statistic), 2), p_value = round(wt$p.value, 4))
})

stat_tests <- bind_rows(chi_results, mw_results) %>%
  mutate(significant = p_value < 0.05) %>%
  arrange(p_value)

print(stat_tests, n = Inf)
write_csv(stat_tests, "outputs/results/statistical_tests.csv")

# 16: significance lollipop plot
stat_tests %>%
  slice_head(n = 20) %>%
  mutate(neg_log_p = -log10(p_value + 1e-10)) %>%
  ggplot(aes(x = reorder(feature, neg_log_p), y = neg_log_p,
             color = significant)) +
  geom_segment(aes(xend = feature, y = 0, yend = neg_log_p), linewidth = 1) +
  geom_point(size = 3) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed",
             color = "red", alpha = 0.7) +
  coord_flip() +
  scale_color_manual(values = c("TRUE" = "#D85A30", "FALSE" = "#999999")) +
  labs(title = "feature significance (-log10 p-value)",
       subtitle = "dashed line = p=0.05 threshold",
       x = "", y = "-log10(p-value)", color = "p < 0.05") +
  theme(legend.position = "bottom")
save_fig("16_significance_plot.png", 8, 7)

cat("\nAll EDA plots saved to outputs/figures/\n")
cat("Statistical tests saved to outputs/results/statistical_tests.csv\n")
