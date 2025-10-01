library(dplyr)
library(xtable)

# Read forms vs default data
forms_data <- read.csv("../tables/forms_vs_default_mixed_effects_corrected.csv")

# Extract significant differences (using FDR corrected p-values)
significant_forms <- forms_data %>%
  # Clean form names
  mutate(
    form_clean = case_when(
      form == "both_negation" ~ "Both Negation",
      form == "complex_emb" ~ "Complex Emb", 
      form == "complex_long" ~ "Complex Long",
      form == "context_negation" ~ "Context Negation",
      form == "target_negation" ~ "Target Negation"
    ),
    model_clean = case_when(
      Model == "Baseline" ~ "Baseline",
      Model == "Remove Expletives" ~ "Rmv. Expletives",
      Model == "Impoverish Determiners" ~ "Impvr. Detrmn.",
      Model == "Remove Articles" ~ "Rmv. Articles",
      Model == "Lemmatize Verbs" ~ "Lemmatize Verbs", 
      Model == "Remove Subject Pronominals" ~ "Rmv. Subject Pronominals"
    ),
    # Use pifont checkmark for significant, empty for non-significant
    checkmark = ifelse(significant_fdr_within == TRUE, "\\ding{51}", "")
  ) %>%
  select(form_clean, model_clean, checkmark)

# Create wide format table manually
forms_list <- c("Complex Long", "Complex Emb", "Context Negation", "Target Negation", "Both Negation")
models_list <- c("Baseline", "Rmv. Expletives", "Impvr. Detrmn.", 
                 "Rmv. Articles", "Lemmatize Verbs", "Rmv. Subject Pronominals")

# Initialize the table with ordered forms
checkmark_table <- data.frame(
  Form = forms_list,
  stringsAsFactors = FALSE
)

# Add columns for each model
for (model in models_list) {
  checkmarks <- c()
  for (form in forms_list) {
    mark <- significant_forms$checkmark[significant_forms$form_clean == form & 
                                       significant_forms$model_clean == model]
    if (length(mark) == 0) mark <- ""
    checkmarks <- c(checkmarks, mark[1])
  }
  checkmark_table[[model]] <- checkmarks
}

# Table is already in the right format with Form column first

# Create LaTeX table
xt <- xtable(checkmark_table)

# Save to file with APA formatting
output_path <- "../tables/latex_tables/forms_vs_default_checkmarks.tex"
print(xt, 
      file = output_path,
      include.rownames = FALSE,
      include.colnames = TRUE,
      sanitize.text.function = function(x) x,
      hline.after = c(-1, 0, nrow(checkmark_table)),
      tabular.environment = "tabular")

# Clean up table and add spacing
content <- readLines(output_path)
start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
content <- content[start_idx:end_idx]

# Add spacing and use makecell for headers
content[1] <- "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{8pt}\\renewcommand{\\theadalign}{cc}\n\\begin{tabular}{lcccccc}"

# Format the header row with makecell for line breaks
header_line <- which(grepl("Form &", content))[1]
content[header_line] <- "Form & Baseline & \\thead{Rmv.\\\\Expletives} & \\thead{Impvr.\\\\Detrmn.} & \\thead{Rmv.\\\\Articles} & \\thead{Lemmatize\\\\Verbs} & \\thead{Rmv. Subject\\\\Pronominals} \\\\"

writeLines(content, output_path)

cat("Created forms vs default checkmarks table: forms_vs_default_checkmarks.tex\n")

# Display the table for verification
print("Forms vs Default Significance Table:")
print(checkmark_table)