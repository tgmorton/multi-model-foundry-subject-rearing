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

# Function to create improved itemgroup margin table
create_improved_itemgroup_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Item Group Pairwise") %>%
    arrange(Analysis.Type, Comparison)
  
  # Create formatted table with icons
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
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)],
                        model_data$Comparison[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 2nd", model_data$Comparison)]),
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 3rd", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("1st vs 3rd", model_data$Comparison)]),
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Person Contrasts" & grepl("2nd vs 3rd", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Person Contrasts" & grepl("2nd vs 3rd", model_data$Comparison)]),
      "",
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("1st Person", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("1st Person", model_data$Comparison)]),
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("2nd Person", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("2nd Person", model_data$Comparison)]),
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Number Contrasts" & grepl("3rd Person", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Number Contrasts" & grepl("3rd Person", model_data$Comparison)]),
      "",
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Control Contrasts"],
                        model_data$Significant[model_data$Analysis.Type == "Control Contrasts"]),
      "",
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Expletive Contrasts"],
                        model_data$Significant[model_data$Analysis.Type == "Expletive Contrasts"]),
      "",
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Topic Shift Contrasts"],
                        model_data$Significant[model_data$Analysis.Type == "Topic Shift Contrasts"])
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llc")
  
  # Save to file with improved formatting
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_itemgroups_improved.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = FALSE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(margin_table)),
        tabular.environment = "tabular")
  
  # Remove table wrapper and clean up
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  
  # Remove header row since tables will be captioned
  # Keep first hline as top border
  
  writeLines(content, output_path)
  cat("Created improved itemgroups table:", output_path, "\n")
}

# Function to create improved form margin table  
create_improved_form_margin_table <- function(model_name) {
  model_data <- data %>%
    filter(Model == model_name, Analysis.Category == "Form Pairwise") %>%
    arrange(Analysis.Type, Comparison)
  
  # Create formatted table with icons
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
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Complex Forms"],
                        model_data$Significant[model_data$Analysis.Type == "Complex Forms"]),
      "",
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Negation Types" & grepl("Context", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Negation Types" & grepl("Context", model_data$Comparison)]),
      format_icon_result(model_data$Effect[model_data$Analysis.Type == "Negation Types" & grepl("Both", model_data$Comparison)],
                        model_data$Significant[model_data$Analysis.Type == "Negation Types" & grepl("Both", model_data$Comparison)])
    )
  )
  
  # Create LaTeX table
  model_safe <- gsub(" ", "", model_name)
  xt <- xtable(margin_table, align = "llc")
  
  # Save to file with improved formatting
  output_path <- paste0("../tables/latex_tables/margin/", tolower(model_safe), "_forms_improved.tex")
  print(xt, 
        file = output_path,
        include.rownames = FALSE,
        include.colnames = FALSE,
        sanitize.text.function = function(x) x,
        hline.after = c(-1, nrow(margin_table)),
        tabular.environment = "tabular")
  
  # Remove table wrapper and clean up
  content <- readLines(output_path)
  start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
  end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
  content <- content[start_idx:end_idx]
  
  # Remove header row since tables will be captioned
  # Keep first hline as top border
  
  writeLines(content, output_path)
  cat("Created improved forms table:", output_path, "\n")
}

# Create comprehensive tables with improved icons
create_comprehensive_improved_tables <- function() {
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
  output_items <- "../tables/latex_tables/margin/comprehensive_itemgroups_improved.tex"
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
  output_forms <- "../tables/latex_tables/margin/comprehensive_forms_improved.tex"
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
  
  cat("Created comprehensive improved tables\n")
}

# Generate all improved tables
models <- unique(data$Model[data$Analysis.Category %in% c("Item Group Pairwise", "Form Pairwise")])

cat("Creating improved margin tables with arrow icons...\n")
cat("Legend: $\\uparrow$ = first group higher, $\\downarrow$ = second group higher, $-$ = no difference\n")
cat("Significance: *** p<0.001, ** p<0.01, * p<0.05\n\n")

for (model in models) {
  cat("Processing model:", model, "\n")
  create_improved_itemgroup_margin_table(model)
  create_improved_form_margin_table(model)
}

create_comprehensive_improved_tables()
cat("\nAll improved margin tables created!\n")