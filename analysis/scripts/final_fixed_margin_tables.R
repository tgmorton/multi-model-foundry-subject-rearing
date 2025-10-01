library(dplyr)
library(xtable)

# Read the comprehensive results
data <- read.csv("../tables/final_comprehensive_mixed_effects_comparisons.csv")

# Simple function to format results based on comparison and effect
format_icon_result <- function(comparison, effect, significant) {
  if (grepl("No difference", effect)) {
    direction_icon <- "$-$"
  } else {
    # Parse comparison to get first and second groups
    if (grepl(":", comparison)) {
      # Handle "1st Person: Singular vs Plural" format
      parts <- strsplit(comparison, ": ")[[1]][2]  # Get "Singular vs Plural"
      parts <- strsplit(parts, " vs ")[[1]]
      first_group <- tolower(trimws(parts[1]))   # "singular"
      second_group <- tolower(trimws(parts[2]))  # "plural"
    } else {
      # Handle "1st vs 2nd Person" or "Target vs Context Negation" format
      parts <- strsplit(comparison, " vs ")[[1]]
      first_group <- tolower(trimws(parts[1]))
      second_group <- tolower(trimws(parts[2]))
      # Remove common suffixes
      first_group <- gsub(" person| negation", "", first_group)
      second_group <- gsub(" person| negation", "", second_group)
    }
    
    # Check if first group is mentioned as higher in effect
    effect_lower <- tolower(effect)
    if (grepl(paste0(first_group, " >"), effect_lower)) {
      direction_icon <- "$\\uparrow$"  # First group higher
    } else if (grepl(paste0(second_group, " >"), effect_lower)) {
      direction_icon <- "$\\downarrow$"  # Second group higher
    } else {
      # Fallback: look for any ">" pattern
      direction_icon <- "$\\uparrow$"  # Default
    }
  }
  
  sig_level <- case_when(
    significant == "***" ~ "***",
    significant == "**" ~ "**", 
    significant == "*" ~ "*",
    TRUE ~ ""
  )
  
  result <- ifelse(direction_icon == "$-$", "$-$", paste0(direction_icon, sig_level))
  return(result)
}

# Helper function to get result for a specific comparison
get_result <- function(model_data, analysis_type, comparison_pattern) {
  row <- model_data %>%
    filter(Analysis.Type == analysis_type & grepl(comparison_pattern, Comparison, fixed = TRUE))
  
  if (nrow(row) == 0) return("")
  
  return(format_icon_result(row$Comparison[1], row$Effect[1], row$Significant[1]))
}

# Function to create improved itemgroup margin table
create_improved_itemgroup_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Item Group Pairwise")
  
  # Create formatted table using helper function
  margin_table <- data.frame(
    Test = c(
      "\\textbf{Person}",
      "\\quad 1st vs 2nd", 
      "\\quad 1st vs 3rd",
      "\\quad 2nd vs 3rd",
      "\\textbf{Number}",
      "\\quad 1st: sg vs pl",
      "\\quad 2nd: sg vs pl", 
      "\\quad 3rd: sg vs pl",
      "\\textbf{Control}",
      "\\quad Subj vs Obj",
      "\\textbf{Expletive}", 
      "\\quad Seems vs Be",
      "\\textbf{Topic}",
      "\\quad No vs Shift"
    ),
    Result = c(
      "",
      get_result(model_data, "Person Contrasts", "1st vs 2nd Person"),
      get_result(model_data, "Person Contrasts", "1st vs 3rd Person"),
      get_result(model_data, "Person Contrasts", "2nd vs 3rd Person"),
      "",
      get_result(model_data, "Number Contrasts", "1st Person: Singular vs Plural"),
      get_result(model_data, "Number Contrasts", "2nd Person: Singular vs Plural"),
      get_result(model_data, "Number Contrasts", "3rd Person: Singular vs Plural"),
      "",
      get_result(model_data, "Control Contrasts", "Subject vs Object Control"),
      "",
      get_result(model_data, "Expletive Contrasts", "Seems vs Be Expletives"),
      "",
      get_result(model_data, "Topic Shift Contrasts", "No Topic Shift vs Topic Shift")
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llc")
  
  # Save to file with improved formatting
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_itemgroups_fixed.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = FALSE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(margin_table)),
        tabular.environment = "tabular")
  
  # Remove table wrapper
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_path)
  
  cat("Created fixed itemgroups table:", output_path, "\n")
}

# Function to create improved form margin table  
create_improved_form_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Form Pairwise")
  
  # Create formatted table using helper function
  margin_table <- data.frame(
    Test = c(
      "\\textbf{Complex}",
      "\\quad Emb vs Long", 
      "\\textbf{Negation}",
      "\\quad Targ vs Cont",
      "\\quad Targ vs Both"
    ),
    Result = c(
      "",
      get_result(model_data, "Complex Forms", "Complex Embedded vs Long"),
      "",
      get_result(model_data, "Negation Types", "Target vs Context Negation"),
      get_result(model_data, "Negation Types", "Target vs Both Negation")
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llc")
  
  # Save to file with improved formatting
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_forms_fixed.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = FALSE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(margin_table)),
        tabular.environment = "tabular")
  
  # Remove table wrapper
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_path)
  
  cat("Created fixed forms table:", output_path, "\n")
}

# Create comprehensive tables with fixed icons
create_comprehensive_fixed_tables <- function() {
  # Get all pairwise data
  itemgroup_data <- data %>%
    filter(Analysis.Category == "Item Group Pairwise") %>%
    select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
    mutate(Result = format_icon_result(Comparison, Effect, Significant))
  
  forms_data <- data %>%
    filter(Analysis.Category == "Form Pairwise") %>%
    select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
    mutate(Result = format_icon_result(Comparison, Effect, Significant))
  
  # Itemgroups comprehensive table
  itemgroup_map <- data.frame(
    Test = c("\\textbf{Person}", "1st vs 2nd", "1st vs 3rd", "2nd vs 3rd",
             "\\textbf{Number}", "1st: sg vs pl", "2nd: sg vs pl", "3rd: sg vs pl",
             "\\textbf{Control}", "Subj vs Obj", "\\textbf{Expletive}", "Seems vs Be",
             "\\textbf{Topic}", "No vs Shift"),
    lookup_type = c("", "Person Contrasts", "Person Contrasts", "Person Contrasts",
                    "", "Number Contrasts", "Number Contrasts", "Number Contrasts",
                    "", "Control Contrasts", "", "Expletive Contrasts", "", "Topic Shift Contrasts"),
    lookup_comparison = c("", "1st vs 2nd Person", "1st vs 3rd Person", "2nd vs 3rd Person",
                         "", "1st Person: Singular vs Plural", "2nd Person: Singular vs Plural", "3rd Person: Singular vs Plural",
                         "", "Subject vs Object Control", "", "Seems vs Be Expletives", "", "No Topic Shift vs Topic Shift")
  )
  
  models <- c("Baseline", "Impoverish Determiners", "Lemmatize Verbs", 
              "Remove Articles", "Remove Expletives", "Remove Subject Pronominals")
  
  comprehensive_itemgroup <- data.frame(Test = itemgroup_map$Test)
  
  for (model in models) {
    model_results <- c()
    for (i in 1:nrow(itemgroup_map)) {
      if (itemgroup_map$lookup_type[i] == "") {
        model_results <- c(model_results, "")
      } else {
        result <- itemgroup_data %>%
          filter(Model == model, 
                 Analysis.Type == itemgroup_map$lookup_type[i],
                 Comparison == itemgroup_map$lookup_comparison[i]) %>%
          pull(Result)
        
        if (length(result) == 0) result <- ""
        model_results <- c(model_results, result[1])
      }
    }
    comprehensive_itemgroup[[gsub(" ", "", model)]] <- model_results
  }
  
  # Forms comprehensive table
  forms_map <- data.frame(
    Test = c("\\textbf{Complex}", "Emb vs Long", "\\textbf{Negation}", "Targ vs Cont", "Targ vs Both"),
    lookup_type = c("", "Complex Forms", "", "Negation Types", "Negation Types"),
    lookup_comparison = c("", "Complex Embedded vs Long", "", "Target vs Context Negation", "Target vs Both Negation")
  )
  
  comprehensive_forms <- data.frame(Test = forms_map$Test)
  
  for (model in models) {
    model_results <- c()
    for (i in 1:nrow(forms_map)) {
      if (forms_map$lookup_type[i] == "") {
        model_results <- c(model_results, "")
      } else {
        result <- forms_data %>%
          filter(Model == model, 
                 Analysis.Type == forms_map$lookup_type[i],
                 Comparison == forms_map$lookup_comparison[i]) %>%
          pull(Result)
        
        if (length(result) == 0) result <- ""
        model_results <- c(model_results, result[1])
      }
    }
    comprehensive_forms[[gsub(" ", "", model)]] <- model_results
  }
  
  # Short model names for headers
  names(comprehensive_itemgroup) <- c("Test", "Base", "Imp", "Lem", "Art", "Exp", "Pron")
  names(comprehensive_forms) <- c("Test", "Base", "Imp", "Lem", "Art", "Exp", "Pron")
  
  # Create itemgroups table
  xt_items <- xtable(comprehensive_itemgroup)
  output_items <- "../tables/latex_tables/margin/comprehensive_itemgroups_fixed.tex"
  print(xt_items, 
        file = output_items,
        include.rownames = FALSE,
        include.colnames = TRUE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(comprehensive_itemgroup)),
        tabular.environment = "tabular")
  
  # Clean up itemgroups table
  content <- readLines(output_items)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_items)
  
  # Create forms table  
  xt_forms <- xtable(comprehensive_forms)
  output_forms <- "../tables/latex_tables/margin/comprehensive_forms_fixed.tex"
  print(xt_forms, 
        file = output_forms,
        include.rownames = FALSE,
        include.colnames = TRUE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(comprehensive_forms)),
        tabular.environment = "tabular")
  
  # Clean up forms table
  content <- readLines(output_forms)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  writeLines(content, output_forms)
  
  cat("Created comprehensive fixed tables\n")
}

# Generate all fixed tables
models <- unique(data$Model[data$Analysis.Category %in% c("Item Group Pairwise", "Form Pairwise")])

cat("Creating fixed margin tables with correct arrow directions...\n")
cat("Legend: $\\uparrow$ = first group higher, $\\downarrow$ = second group higher, $-$ = no difference\n")
cat("Significance: *** p<0.001, ** p<0.01, * p<0.05\n\n")

for (model in models) {
  cat("Processing model:", model, "\n")
  create_improved_itemgroup_margin_table(model)
  create_improved_form_margin_table(model)
}

create_comprehensive_fixed_tables()
cat("\nAll fixed margin tables created!\n")

# Test specific cases to verify arrows are correct
cat("\nVerification of arrow directions:\n")
test_baseline <- data %>%
  filter(Model == "Baseline", Analysis.Type == "Number Contrasts") %>%
  select(Comparison, Effect, Significant) %>%
  rowwise() %>%
  mutate(Result = format_icon_result(Comparison, Effect, Significant)) %>%
  ungroup()

print(test_baseline)