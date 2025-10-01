library(dplyr)
library(xtable)

# Read the comprehensive results
data <- read.csv("../tables/final_comprehensive_mixed_effects_comparisons.csv")

# Function to format result and significance together
format_result_sig <- function(effect, significant) {
  direction <- case_when(
    grepl(" > ", effect) ~ ">",
    grepl(" < ", effect) ~ "<", 
    grepl("No difference", effect) ~ "="
  )
  
  sig <- case_when(
    significant == "***" ~ "***",
    significant == "**" ~ "**", 
    significant == "*" ~ "*",
    TRUE ~ ""
  )
  
  paste0(direction, sig)
}

# Create comprehensive itemgroup comparison table
create_comprehensive_itemgroup_table <- function() {
  # Get pairwise data for all models
  itemgroup_data <- data %>%
    filter(Analysis.Category == "Item Group Pairwise") %>%
    select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
    mutate(Result = format_result_sig(Effect, Significant))
  
  # Create comparison mapping
  comparison_map <- data.frame(
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
    lookup_type = c(
      "",
      "Person Contrasts",
      "Person Contrasts",
      "Person Contrasts",
      "",
      "Number Contrasts",
      "Number Contrasts",
      "Number Contrasts",
      "",
      "Control Contrasts",
      "",
      "Expletive Contrasts",
      "",
      "Topic Shift Contrasts"
    ),
    lookup_comparison = c(
      "",
      "1st vs 2nd Person",
      "1st vs 3rd Person", 
      "2nd vs 3rd Person",
      "",
      "1st Person: Singular vs Plural",
      "2nd Person: Singular vs Plural",
      "3rd Person: Singular vs Plural",
      "",
      "Subject vs Object Control",
      "",
      "Seems vs Be Expletives",
      "",
      "No Topic Shift vs Topic Shift"
    )
  )
  
  # Get models in order
  models <- c("Baseline", "Impoverish Determiners", "Lemmatize Verbs", 
              "Remove Articles", "Remove Expletives", "Remove Subject Pronominals")
  
  # Create the comprehensive table - start with just Test column
  comprehensive_table <- data.frame(Test = comparison_map$Test)
  
  for (model in models) {
    model_results <- c()
    for (i in 1:nrow(comparison_map)) {
      if (comparison_map$lookup_type[i] == "") {
        model_results <- c(model_results, "")
      } else {
        result <- itemgroup_data %>%
          filter(Model == model, 
                 Analysis.Type == comparison_map$lookup_type[i],
                 Comparison == comparison_map$lookup_comparison[i]) %>%
          pull(Result)
        
        if (length(result) == 0) result <- ""
        model_results <- c(model_results, result[1])
      }
    }
    comprehensive_table[[gsub(" ", "", model)]] <- model_results
  }
  
  # Create LaTeX table with shorter model names  
  model_names <- c("Test", "Base", "Imp", "Lem", "Art", "Exp", "Pron")
  names(comprehensive_table) <- model_names[1:ncol(comprehensive_table)]
  
  xt <- xtable(comprehensive_table)
  
  # Save to file
  output_path <- "../tables/latex_tables/margin/comprehensive_itemgroups_comparison.tex"
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
  
  cat("Created comprehensive itemgroups table:", output_path, "\n")
}

# Create comprehensive forms comparison table
create_comprehensive_forms_table <- function() {
  # Get pairwise data for all models
  forms_data <- data %>%
    filter(Analysis.Category == "Form Pairwise") %>%
    select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
    mutate(Result = format_result_sig(Effect, Significant))
  
  # Create comparison mapping
  comparison_map <- data.frame(
    Test = c(
      "\\textbf{Complex}",
      "Emb vs Long", 
      "\\textbf{Negation}",
      "Targ vs Cont",
      "Targ vs Both"
    ),
    lookup_type = c(
      "",
      "Complex Forms",
      "",
      "Negation Types",
      "Negation Types"
    ),
    lookup_comparison = c(
      "",
      "Complex Embedded vs Long",
      "",
      "Target vs Context Negation",
      "Target vs Both Negation"
    )
  )
  
  # Get models in order
  models <- c("Baseline", "Impoverish Determiners", "Lemmatize Verbs", 
              "Remove Articles", "Remove Expletives", "Remove Subject Pronominals")
  
  # Create the comprehensive table - start with just Test column
  comprehensive_table <- data.frame(Test = comparison_map$Test)
  
  for (model in models) {
    model_results <- c()
    for (i in 1:nrow(comparison_map)) {
      if (comparison_map$lookup_type[i] == "") {
        model_results <- c(model_results, "")
      } else {
        result <- forms_data %>%
          filter(Model == model, 
                 Analysis.Type == comparison_map$lookup_type[i],
                 Comparison == comparison_map$lookup_comparison[i]) %>%
          pull(Result)
        
        if (length(result) == 0) result <- ""
        model_results <- c(model_results, result[1])
      }
    }
    comprehensive_table[[gsub(" ", "", model)]] <- model_results
  }
  
  # Create LaTeX table with shorter model names  
  model_names <- c("Test", "Base", "Imp", "Lem", "Art", "Exp", "Pron")
  names(comprehensive_table) <- model_names[1:ncol(comprehensive_table)]
  
  xt <- xtable(comprehensive_table)
  
  # Save to file
  output_path <- "../tables/latex_tables/margin/comprehensive_forms_comparison.tex"
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
  
  cat("Created comprehensive forms table:", output_path, "\n")
}

# Generate comprehensive tables
create_comprehensive_itemgroup_table()
create_comprehensive_forms_table()

cat("All comprehensive comparison tables created!\n")