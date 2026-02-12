library(dplyr)
library(xtable)

# Read the comprehensive results
data <- read.csv("../tables/final_comprehensive_mixed_effects_comparisons.csv")

# Function to format significance levels
format_sig <- function(p_corrected, significant) {
  case_when(
    significant == "***" ~ "***",
    significant == "**" ~ "**", 
    significant == "*" ~ "*",
    TRUE ~ ""
  )
}

# Function to format comparison direction
format_direction <- function(effect) {
  case_when(
    grepl(" > ", effect) ~ ">",
    grepl(" < ", effect) ~ "<", 
    grepl("No difference", effect) ~ "="
  )
}

# Function to create margin table for itemgroups
create_itemgroup_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Item Group Pairwise") %>%
    arrange(Analysis.Type, Comparison)
  
  # Create formatted table
  margin_table <- data.frame(
    Test = c(
      "\\textbf{Person}",
      "1st vs 2nd", 
      "1st vs 3rd",
      "2nd vs 3rd",
      "\\textbf{Number}",
      "1st: Sing vs Plur",
      "2nd: Sing vs Plur", 
      "3rd: Sing vs Plur",
      "\\textbf{Control}",
      "Subj vs Obj",
      "\\textbf{Expletive}", 
      "Seems vs Be",
      "\\textbf{Topic}",
      "No Shift vs Shift"
    ),
    Result = c(
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)]),
      format_direction(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 3rd", model_data$Comparison)]),
      format_direction(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("2nd vs 3rd", model_data$Comparison)]),
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("1st Person", model_data$Comparison)]),
      format_direction(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("2nd Person", model_data$Comparison)]),
      format_direction(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("3rd Person", model_data$Comparison)]),
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Control Contrasts"]),
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Expletive Contrasts"]),
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Topic Shift Contrasts"])
    ),
    Sig = c(
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)]),
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 3rd", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 3rd", model_data$Comparison)]),
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Person Contrasts" & grepl("2nd vs 3rd", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("2nd vs 3rd", model_data$Comparison)]),
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Number Contrasts" & grepl("1st Person", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("1st Person", model_data$Comparison)]),
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Number Contrasts" & grepl("2nd Person", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("2nd Person", model_data$Comparison)]),
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Number Contrasts" & grepl("3rd Person", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("3rd Person", model_data$Comparison)]),
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Control Contrasts"], 
                 model_data$Significant[model_data$Analysis.Type == "Control Contrasts"]),
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Expletive Contrasts"], 
                 model_data$Significant[model_data$Analysis.Type == "Expletive Contrasts"]),
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Topic Shift Contrasts"], 
                 model_data$Significant[model_data$Analysis.Type == "Topic Shift Contrasts"])
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llcc")
  
  # Save to file
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_itemgroups_margin.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = TRUE,
        sanitize.text.function = function(x) x,
        hline.after = NULL,
        add.to.row = list(pos = list(0), command = "\\hline "),
        tabular.environment = "tabular")
  
  # Remove table wrapper
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_path)
  
  cat("Created:", output_path, "\n")
}

# Function to create margin table for forms  
create_form_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Form Pairwise") %>%
    arrange(Analysis.Type, Comparison)
  
  # Create formatted table
  margin_table <- data.frame(
    Test = c(
      "\\textbf{Complex}",
      "Emb vs Long", 
      "\\textbf{Negation}",
      "Targ vs Cont",
      "Targ vs Both"
    ),
    Result = c(
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Complex Forms"]),
      "",
      format_direction(model_data$Effect[model_data$Analysis.Type == "Negation Types" & grepl("Context", model_data$Comparison)]),
      format_direction(model_data$Effect[model_data$Analysis.Type == "Negation Types" & grepl("Both", model_data$Comparison)])
    ),
    Sig = c(
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Complex Forms"], 
                 model_data$Significant[model_data$Analysis.Type == "Complex Forms"]),
      "",
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Negation Types" & grepl("Context", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Negation Types" & grepl("Context", model_data$Comparison)]),
      format_sig(model_data$p.corrected[model_data$Analysis.Type == "Negation Types" & grepl("Both", model_data$Comparison)], 
                 model_data$Significant[model_data$Analysis.Type == "Negation Types" & grepl("Both", model_data$Comparison)])
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llcc")
  
  # Save to file
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_forms_margin.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = TRUE,
        sanitize.text.function = function(x) x,
        hline.after = NULL,
        add.to.row = list(pos = list(0), command = "\\hline "),
        tabular.environment = "tabular")
  
  # Remove table wrapper
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_path)
  
  cat("Created:", output_path, "\n")
}

# Create directory if it doesn't exist
dir.create("../tables/latex_tables/margin", showWarnings = FALSE, recursive = TRUE)

# Get unique models
models <- unique(data$Model[data$Analysis.Category %in% c("Item Group Pairwise", "Form Pairwise")])

# Generate tables for each model
for (model in models) {
  cat("Processing model:", model, "\n")
  create_itemgroup_margin_table(model)
  create_form_margin_table(model)
}

cat("All margin tables created!\n")