library(dplyr)
library(xtable)

# Read the comprehensive results
data <- read.csv("../tables/final_comprehensive_mixed_effects_comparisons.csv")

# Simple vectorized function
format_icon_result_vectorized <- function(comparison_vec, effect_vec, significant_vec) {
  results <- character(length(comparison_vec))
  
  for (i in seq_along(comparison_vec)) {
    comparison <- comparison_vec[i]
    effect <- effect_vec[i]
    significant <- significant_vec[i]
    
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
    
    results[i] <- ifelse(direction_icon == "$-$", "$-$", paste0(direction_icon, sig_level))
  }
  
  return(results)
}

# Create comprehensive tables with fixed icons
itemgroup_data <- data %>%
  filter(Analysis.Category == "Item Group Pairwise") %>%
  select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
  mutate(Result = format_icon_result_vectorized(Comparison, Effect, Significant))

forms_data <- data %>%
  filter(Analysis.Category == "Form Pairwise") %>%
  select(Model, Analysis.Type, Comparison, Effect, Significant) %>%
  mutate(Result = format_icon_result_vectorized(Comparison, Effect, Significant))

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

models <- c("Baseline", "Remove Expletives", "Impoverish Determiners", 
            "Remove Articles", "Lemmatize Verbs", "Remove Subject Pronominals")

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

# Full model names for headers with line breaks 
names(comprehensive_itemgroup) <- c("Test", "Baseline", "Rmv. Expletives", "Impvr. Detrmn.", "Rmv. Articles", "Lemmatize Verbs", "Rmv. Subject Pronominals")
names(comprehensive_forms) <- c("Test", "Baseline", "Rmv. Expletives", "Impvr. Detrmn.", "Rmv. Articles", "Lemmatize Verbs", "Rmv. Subject Pronominals")

# Create itemgroups table
xt_items <- xtable(comprehensive_itemgroup)
output_items <- "../tables/latex_tables/margin/comprehensive_itemgroups_fixed.tex"
print(xt_items, 
      file = output_items,
      include.rownames = FALSE,
      include.colnames = TRUE,
      sanitize.text.function = function(x) x,
      hline.after = c(-1, 0, nrow(comprehensive_itemgroup)),
      tabular.environment = "tabular")

# Clean up itemgroups table and add spacing
content <- readLines(output_items)
start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
content <- content[start_idx:end_idx]
# Add moderate spacing and makecell package setup
content[1] <- "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{8pt}\\renewcommand{\\theadalign}{cc}\n\\begin{tabular}{lcccccc}"
# Replace the header row with makecell \\thead commands
header_line <- which(grepl("Test & Baseline", content))[1]
content[header_line] <- "Test & Baseline & \\thead{Rmv.\\\\Expletives} & \\thead{Impvr.\\\\Detrmn.} & \\thead{Rmv.\\\\Articles} & \\thead{Lemmatize\\\\Verbs} & \\thead{Rmv. Subject\\\\Pronominals} \\\\"
writeLines(content, output_items)

# Create forms table  
xt_forms <- xtable(comprehensive_forms)
output_forms <- "../tables/latex_tables/margin/comprehensive_forms_fixed.tex"
print(xt_forms, 
      file = output_forms,
      include.rownames = FALSE,
      include.colnames = TRUE,
      sanitize.text.function = function(x) x,
      hline.after = c(-1, 0, nrow(comprehensive_forms)),
      tabular.environment = "tabular")

# Clean up forms table and add spacing
content <- readLines(output_forms)
start_idx <- which(grepl("\\\\begin\\{tabular\\}", content))
end_idx <- which(grepl("\\\\end\\{tabular\\}", content))
content <- content[start_idx:end_idx]
# Add moderate spacing and makecell package setup
content[1] <- "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{8pt}\\renewcommand{\\theadalign}{cc}\n\\begin{tabular}{lcccccc}"
# Replace the header row with makecell \\thead commands
header_line <- which(grepl("Test & Baseline", content))[1]
content[header_line] <- "Test & Baseline & \\thead{Rmv.\\\\Expletives} & \\thead{Impvr.\\\\Detrmn.} & \\thead{Rmv.\\\\Articles} & \\thead{Lemmatize\\\\Verbs} & \\thead{Rmv. Subject\\\\Pronominals} \\\\"
writeLines(content, output_forms)

cat("Created comprehensive fixed tables:\n")
cat("- comprehensive_itemgroups_fixed.tex\n")
cat("- comprehensive_forms_fixed.tex\n")

# Verify one key pattern
cat("\nVerification - Baseline 2nd person should show downward arrow:\n")
baseline_2nd <- comprehensive_itemgroup %>%
  filter(Test == "2nd: sg vs pl") %>%
  select(Test, Baseline)
print(baseline_2nd)