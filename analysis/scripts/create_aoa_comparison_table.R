library(dplyr)
library(xtable)

# Read AOA data
aoa_data <- read.csv("../tables/tests/aoa_halfway_by_model.csv")
delta_data <- read.csv("../tables/tests/delta_aoa_half_vs_baseline.csv")

# Clean and format the data
aoa_comparison <- aoa_data %>%
  # Round to whole numbers for readability
  mutate(
    AOA = round(t_half_checkpoint),
    CI_Lower = round(CI_lo),
    CI_Upper = round(CI_hi),
    CI_formatted = paste0("[", CI_Lower, ", ", 
                         ifelse(CI_Upper > 5000, ">5000", CI_Upper), "]")
  ) %>%
  select(Model, AOA, CI_formatted)

# Add delta information
delta_formatted <- delta_data %>%
  mutate(
    Delta = round(delta),
    Delta_CI = paste0("[", round(d_lo), ", ", 
                     ifelse(d_hi > 5000, ">5000", round(d_hi)), "]"),
    Significance = case_when(
      p_emp < 0.001 ~ "***",
      p_emp < 0.01 ~ "**", 
      p_emp < 0.05 ~ "*",
      TRUE ~ ""
    ),
    Delta_formatted = paste0(ifelse(Delta > 0, "+", ""), Delta, Significance)
  ) %>%
  select(Model, Delta_formatted, Delta_CI)

# Combine the data
final_table <- aoa_comparison %>%
  left_join(delta_formatted, by = "Model") %>%
  # Replace NA values for Baseline with dashes
  mutate(
    Delta_formatted = ifelse(Model == "Baseline", "—", Delta_formatted),
    Delta_CI = ifelse(Model == "Baseline", "—", Delta_CI)
  ) %>%
  # Order models appropriately
  mutate(Model_order = case_when(
    Model == "Baseline" ~ 1,
    Model == "Remove Expletives" ~ 2,
    Model == "Impoverish Determiners" ~ 3,
    Model == "Remove Articles" ~ 4,
    Model == "Lemmatize Verbs" ~ 5,
    Model == "Remove Subject Pronominals" ~ 6
  )) %>%
  arrange(Model_order) %>%
  select(-Model_order)

# Abbreviate model names for table
final_table$Model <- case_when(
  final_table$Model == "Baseline" ~ "Baseline",
  final_table$Model == "Remove Expletives" ~ "Rmv. Expletives",
  final_table$Model == "Impoverish Determiners" ~ "Impvr. Detrmn.",
  final_table$Model == "Remove Articles" ~ "Rmv. Articles", 
  final_table$Model == "Lemmatize Verbs" ~ "Lemmatize Verbs",
  final_table$Model == "Remove Subject Pronominals" ~ "Rmv. Subject Pronominals"
)

# Create proper column names
names(final_table) <- c("Model", "AOA", "95\\% CI", "\\Delta vs Baseline", "\\Delta 95\\% CI")

# Create LaTeX table
xt <- xtable(final_table)

# Save to file with APA formatting
output_path <- "../tables/latex_tables/aoa_comparison_table.tex"
print(xt, 
      file = output_path,
      include.rownames = FALSE,
      include.colnames = TRUE,
      sanitize.text.function = function(x) x,
      hline.after = c(-1, 0, nrow(final_table)),
      tabular.environment = "tabular")

# Clean up table and add spacing
content <- readLines(output_path)
start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
content <- content[start_idx:end_idx]

# Add spacing and use makecell for headers
content[1] <- "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{8pt}\\renewcommand{\\theadalign}{cc}\n\\begin{tabular}{lcccc}"

# Format the header row with makecell (using Unicode delta for LuaTeX)
header_line <- which(grepl("Model &", content))[1]
content[header_line] <- "Model & AOA & \\thead{95\\\\\\% CI} & \\thead{Δ vs\\\\\\\\Baseline} & \\thead{Δ 95\\\\\\% CI} \\\\"

writeLines(content, output_path)

cat("Created AOA comparison table: aoa_comparison_table.tex\n")

# Display the table for verification
print("AOA Comparison Table:")
print(final_table)