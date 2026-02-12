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

# Test the results
cat("Testing Baseline Number Contrasts:\n")
baseline_test <- itemgroup_data %>%
  filter(Model == "Baseline", Analysis.Type == "Number Contrasts") %>%
  select(Comparison, Effect, Result)
print(baseline_test)

cat("\nAll comprehensive data processed successfully!\n")