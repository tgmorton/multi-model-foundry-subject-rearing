# Apply Multiple Comparisons Corrections to Item Group Analyses
# =============================================================

library(tidyverse)

cat("=== CORRECTING ITEM GROUP COMPARISONS ===\n\n")

# List of item group pairwise files
pairwise_files <- list(
  "analysis/tables/pairwise_b1_number_contrasts.csv",
  "analysis/tables/pairwise_b2_person_contrasts.csv", 
  "analysis/tables/pairwise_b3_control_contrasts.csv",
  "analysis/tables/pairwise_b4_expletive_contrasts.csv",
  "analysis/tables/pairwise_b5_topic_shift_contrasts.csv",
  "analysis/tables/pairwise_c1_forms_vs_default.csv",
  "analysis/tables/pairwise_c2_complex_embedding.csv",
  "analysis/tables/pairwise_c3_negation_types.csv"
)

# Initialize storage for all p-values across analyses
all_pvalues <- tibble()
corrected_files <- list()

# Read all p-values first to determine total number of tests
cat("Reading all item group comparison files...\n")
for (i in seq_along(pairwise_files)) {
  file_path <- pairwise_files[[i]]
  if (file.exists(file_path)) {
    cat(sprintf("  %s\n", basename(file_path)))
    
    data <- read_csv(file_path, show_col_types = FALSE)
    
    # Check if p_value column exists
    if ("p_value" %in% names(data)) {
      file_pvalues <- data %>%
        mutate(
          file_name = basename(file_path),
          analysis_group = str_extract(basename(file_path), "^[^_]+_[^_]+"),
          row_id = row_number()
        ) %>%
        select(file_name, analysis_group, row_id, p_value, everything())
      
      all_pvalues <- bind_rows(all_pvalues, file_pvalues)
    }
  }
}

total_tests <- nrow(all_pvalues)
cat(sprintf("\nTotal tests across all item group analyses: %d\n\n", total_tests))

# Apply FDR correction across ALL item group tests
if (total_tests > 0) {
  all_pvalues <- all_pvalues %>%
    mutate(
      p_value_fdr_global = p.adjust(p_value, method = "fdr"),
      p_value_bonferroni_global = p.adjust(p_value, method = "bonferroni"),
      significant_uncorrected = p_value < 0.05,
      significant_fdr_global = p_value_fdr_global < 0.05,
      significant_bonferroni_global = p_value_bonferroni_global < 0.05
    )
  
  # Also apply within-analysis FDR correction
  all_pvalues <- all_pvalues %>%
    group_by(analysis_group) %>%
    mutate(
      p_value_fdr_within = p.adjust(p_value, method = "fdr"),
      significant_fdr_within = p_value_fdr_within < 0.05
    ) %>%
    ungroup()
  
  # Save corrected data back to individual files
  cat("Applying corrections and saving files...\n")
  for (i in seq_along(pairwise_files)) {
    file_path <- pairwise_files[[i]]
    file_name <- basename(file_path)
    
    if (file.exists(file_path)) {
      # Get corrected data for this file
      file_corrected <- all_pvalues %>%
        filter(file_name == !!file_name) %>%
        select(-file_name, -analysis_group, -row_id)
      
      # Save corrected version
      corrected_path <- str_replace(file_path, "\\.csv$", "_corrected.csv")
      write_csv(file_corrected, corrected_path)
      
      cat(sprintf("  %s → %s\n", file_name, basename(corrected_path)))
      
      # Store for summary
      corrected_files[[file_name]] <- file_corrected
    }
  }
  
  # SUMMARY STATISTICS
  cat("\n=== CORRECTION SUMMARY ===\n")
  
  summary_stats <- all_pvalues %>%
    group_by(analysis_group) %>%
    summarise(
      n_tests = n(),
      significant_uncorrected = sum(significant_uncorrected),
      significant_fdr_within = sum(significant_fdr_within),
      significant_fdr_global = sum(significant_fdr_global),
      significant_bonferroni = sum(significant_bonferroni_global),
      .groups = "drop"
    )
  
  print(summary_stats)
  
  # Overall summary
  overall_summary <- all_pvalues %>%
    summarise(
      total_tests = n(),
      significant_uncorrected = sum(significant_uncorrected),
      significant_fdr_global = sum(significant_fdr_global),
      significant_bonferroni = sum(significant_bonferroni_global),
      .groups = "drop"
    )
  
  cat("\nOverall Summary:\n")
  cat(sprintf("Total tests: %d\n", overall_summary$total_tests))
  cat(sprintf("Significant (uncorrected): %d (%.1f%%)\n", 
              overall_summary$significant_uncorrected,
              100 * overall_summary$significant_uncorrected / overall_summary$total_tests))
  cat(sprintf("Significant (FDR corrected): %d (%.1f%%)\n", 
              overall_summary$significant_fdr_global,
              100 * overall_summary$significant_fdr_global / overall_summary$total_tests))
  cat(sprintf("Significant (Bonferroni corrected): %d (%.1f%%)\n", 
              overall_summary$significant_bonferroni,
              100 * overall_summary$significant_bonferroni / overall_summary$total_tests))
  
  # Show which results changed
  cat("\n=== SIGNIFICANCE CHANGES ===\n")
  changes <- all_pvalues %>%
    filter(significant_uncorrected != significant_fdr_global) %>%
    select(analysis_group, model, p_value, p_value_fdr_global, 
           significant_uncorrected, significant_fdr_global) %>%
    mutate(
      change_type = case_when(
        significant_uncorrected & !significant_fdr_global ~ "Lost significance",
        !significant_uncorrected & significant_fdr_global ~ "Gained significance",
        TRUE ~ "No change"
      )
    )
  
  if (nrow(changes) > 0) {
    cat(sprintf("Number of results that changed significance: %d\n", nrow(changes)))
    print(changes)
  } else {
    cat("No changes in significance after FDR correction\n")
  }
  
  # Save overall corrected dataset
  write_csv(all_pvalues, "analysis/tables/all_item_group_comparisons_corrected.csv")
  write_csv(summary_stats, "analysis/tables/item_group_corrections_summary.csv")
  
  cat("\nSaved comprehensive results to:\n")
  cat("  - analysis/tables/all_item_group_comparisons_corrected.csv (all data)\n")
  cat("  - analysis/tables/item_group_corrections_summary.csv (summary)\n")
  cat("  - Individual *_corrected.csv files for each analysis\n")
  
} else {
  cat("No p-values found in item group files\n")
}

cat("\n=== ITEM GROUP CORRECTIONS COMPLETE ===\n")
cat("Recommendation: Use FDR-corrected p-values for all item group reporting\n")