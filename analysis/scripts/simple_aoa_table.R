library(dplyr)
library(xtable)

# Read AOA data
aoa_data <- read.csv("../tables/tests/aoa_halfway_by_model.csv")

# Create simple table with Model, AOA, and CI, sorted by AOA
simple_aoa <- aoa_data %>%
  # Round values for readability
  mutate(
    AOA = round(t_half_checkpoint),
    CI_Lower = round(CI_lo),
    CI_Upper = round(CI_hi),
    CI = paste0("[", CI_Lower, ", ", 
               ifelse(CI_Upper > 5000, ">5000", CI_Upper), "]")
  ) %>%
  select(Model, AOA, CI) %>%
  # Sort by AOA (fastest acquisition first)
  arrange(AOA)

# Abbreviate model names consistently with other tables
simple_aoa$Model <- case_when(
  simple_aoa$Model == "Baseline" ~ "Baseline",
  simple_aoa$Model == "Remove Expletives" ~ "Rmv. Expletives",
  simple_aoa$Model == "Impoverish Determiners" ~ "Impvr. Detrmn.",
  simple_aoa$Model == "Remove Articles" ~ "Rmv. Articles", 
  simple_aoa$Model == "Lemmatize Verbs" ~ "Lemmatize Verbs",
  simple_aoa$Model == "Remove Subject Pronominals" ~ "Rmv. Subject Pronominals"
)

# Create LaTeX table
xt <- xtable(simple_aoa)

# Save to file with APA formatting
output_path <- "../tables/latex_tables/simple_aoa_table.tex"
print(xt, 
      file = output_path,
      include.rownames = FALSE,
      include.colnames = TRUE,
      sanitize.text.function = function(x) x,
      hline.after = c(-1, 0, nrow(simple_aoa)),
      tabular.environment = "tabular")

# Clean up table and add spacing
content <- readLines(output_path)
start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
content <- content[start_idx:end_idx]

# Add spacing for readability
content[1] <- "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{12pt}\n\\begin{tabular}{lcc}"

writeLines(content, output_path)

cat("Created simple AOA table: simple_aoa_table.tex\n")

# Display the table for verification
print("Simple AOA Table (sorted by acquisition speed):")
print(simple_aoa)