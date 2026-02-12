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
    # Examples: "1st vs 2nd Person", "1st Person: Singular vs Plural", "Target vs Context Negation"
    
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

# Test the function with known cases
test_cases <- data %>%
  filter(Model == "Baseline", Analysis.Category == "Item Group Pairwise") %>%
  select(Comparison, Effect, Significant) %>%
  rowwise() %>%
  mutate(Result = format_icon_result(Comparison, Effect, Significant)) %>%
  ungroup()

print("Testing arrow logic:")
print(test_cases[, c("Comparison", "Effect", "Result")])

# Test with Remove Subject Pronominals number contrasts
test_cases2 <- data %>%
  filter(Model == "Remove Subject Pronominals", Analysis.Type == "Number Contrasts") %>%
  select(Comparison, Effect, Significant) %>%
  rowwise() %>%
  mutate(Result = format_icon_result(Comparison, Effect, Significant)) %>%
  ungroup()

print("\nRemove Subject Pronominals Number Contrasts:")
print(test_cases2[, c("Comparison", "Effect", "Result")])