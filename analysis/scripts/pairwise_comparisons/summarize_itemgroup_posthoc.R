# Summarize Within-Model Item Group Post-hoc Results
# ===================================================

library(tidyverse)
library(knitr)

# Load the pairwise comparisons
pairwise_df <- read_csv("analysis/tables/within_model_itemgroup_pairwise.csv")
emmeans_df <- read_csv("analysis/tables/within_model_itemgroup_emmeans.csv")

cat("\n===========================================\n")
cat("WITHIN-MODEL ITEM GROUP POST-HOC ANALYSIS\n")
cat("===========================================\n\n")

# For each model, show significant differences
models <- unique(pairwise_df$Model)

for (model_name in models) {
  cat(sprintf("\n%s\n", model_name))
  cat(paste(rep("-", nchar(model_name)), collapse=""), "\n\n")
  
  # Get significant comparisons for this model
  sig_pairs <- pairwise_df %>%
    filter(Model == model_name, p.value < 0.05) %>%
    arrange(p.value) %>%
    mutate(
      direction = ifelse(odds.ratio > 1, ">", "<"),
      interpretation = sprintf("%s %s %s", 
                               sub(" / .*", "", contrast),
                               direction,
                               sub(".* / ", "", contrast)),
      effect_size = case_when(
        abs(log(odds.ratio)) > log(3) ~ "large",
        abs(log(odds.ratio)) > log(1.5) ~ "medium",
        TRUE ~ "small"
      )
    )
  
  cat(sprintf("  %d significant pairwise differences (p < 0.05 after FDR correction)\n\n", nrow(sig_pairs)))
  
  if (nrow(sig_pairs) > 0) {
    # Show top 10 most significant
    cat("  Top differences by p-value:\n")
    top_diffs <- sig_pairs %>%
      head(10) %>%
      select(interpretation, odds.ratio, p.value, effect_size) %>%
      mutate(
        OR = sprintf("%.2f", odds.ratio),
        p = format.pval(p.value, digits = 3)
      ) %>%
      select(Comparison = interpretation, OR, `p-value` = p, Effect = effect_size)
    
    print(top_diffs, row.names = FALSE)
    
    # Theoretical contrasts of interest
    cat("\n  Key theoretical contrasts:\n")
    
    theoretical_pairs <- sig_pairs %>%
      filter(
        grepl("3rdSG.*3rdPL|3rdPL.*3rdSG", contrast) |  # Person number
        grepl("2ndSG.*2ndPL|2ndPL.*2ndSG", contrast) |
        grepl("1stSg.*1stPL|1stPL.*1stSg", contrast) |
        grepl("subject_control.*object_control|object_control.*subject_control", contrast) |  # Control
        grepl("expletive_seems.*expletive_be|expletive_be.*expletive_seems", contrast) |  # Expletives
        grepl("no_topic_shift.*topic_shift|topic_shift.*no_topic_shift", contrast)  # Topic shift
      ) %>%
      select(interpretation, odds.ratio, p.value) %>%
      mutate(
        OR = sprintf("%.2f", odds.ratio),
        p = format.pval(p.value, digits = 3),
        sig = case_when(
          p.value < 0.001 ~ "***",
          p.value < 0.01 ~ "**",
          p.value < 0.05 ~ "*",
          TRUE ~ ""
        )
      ) %>%
      select(Comparison = interpretation, OR, p, sig)
    
    if (nrow(theoretical_pairs) > 0) {
      print(theoretical_pairs, row.names = FALSE)
    } else {
      cat("    No significant theoretical contrasts\n")
    }
  }
  
  # Show item group ranking
  cat("\n  Item group ranking (by null preference):\n")
  ranking <- emmeans_df %>%
    filter(Model == model_name) %>%
    arrange(desc(prob)) %>%
    mutate(
      rank = row_number(),
      pct = sprintf("%.1f%%", prob * 100)
    ) %>%
    select(Rank = rank, `Item Group` = item_group, `Null Pref` = pct)
  
  print(ranking, row.names = FALSE)
}

# Create cross-model comparison of specific contrasts
cat("\n\n===========================================\n")
cat("CROSS-MODEL COMPARISON OF KEY CONTRASTS\n")
cat("===========================================\n\n")

# Select important contrasts
key_patterns <- c(
  "4a_subject_control.*4b_object_control|4b_object_control.*4a_subject_control",
  "1a_3rdSG.*3a_1stSg|3a_1stSg.*1a_3rdSG",
  "5a_expletive_seems.*5b_expletive_be|5b_expletive_be.*5a_expletive_seems"
)

for (pattern in key_patterns) {
  contrast_data <- pairwise_df %>%
    filter(grepl(pattern, contrast)) %>%
    select(Model, contrast, odds.ratio, p.value) %>%
    mutate(
      sig = case_when(
        p.value < 0.001 ~ "***",
        p.value < 0.01 ~ "**",
        p.value < 0.05 ~ "*",
        TRUE ~ "ns"
      ),
      OR_sig = sprintf("%.2f%s", odds.ratio, sig)
    ) %>%
    select(Model, contrast, OR_sig) %>%
    pivot_wider(names_from = Model, values_from = OR_sig)
  
  if (nrow(contrast_data) > 0) {
    cat(sprintf("\n%s:\n", contrast_data$contrast[1]))
    print(contrast_data %>% select(-contrast), row.names = FALSE)
  }
}

cat("\n\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant\n")
cat("\nAnalysis complete!\n")