# Complete Null Subject Acquisition Analysis Pipeline
# ===================================================
# 
# This script runs the full analysis pipeline in the correct order.
# Total runtime: ~15-30 minutes depending on system performance.
#
# Generated files: ~25 CSV tables + comprehensive reports
# 
# ===================================================

cat("==========================================================\n")
cat("COMPLETE NULL SUBJECT ACQUISITION ANALYSIS PIPELINE\n")
cat("==========================================================\n\n")

# Record start time
start_time <- Sys.time()
cat("Analysis started at:", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n\n")

# Set up error handling
options(warn = 1)  # Print warnings as they occur

# Function to run script with error handling
run_script <- function(script_path, description) {
  cat("----------------------------------------------------------\n")
  cat("RUNNING:", description, "\n")
  cat("Script:", script_path, "\n")
  cat("----------------------------------------------------------\n")
  
  script_start <- Sys.time()
  
  tryCatch({
    source(script_path)
    script_end <- Sys.time()
    runtime <- round(as.numeric(difftime(script_end, script_start, units = "mins")), 2)
    cat("\n✓ COMPLETED:", description, "(", runtime, "minutes )\n\n")
  }, error = function(e) {
    cat("\n✗ ERROR in", description, ":\n")
    cat("Error message:", conditionMessage(e), "\n")
    cat("Continuing with next script...\n\n")
  })
}

# ===================================================
# SCRIPT 1: MAIN ACQUISITION ANALYSIS
# ===================================================
run_script(
  "analysis/scripts/analysis_with_models.R",
  "Main acquisition analysis (t50, AoA, end-state performance)"
)

# ===================================================
# SCRIPT 2: FIRST EPOCH LEARNING ASSESSMENT
# ===================================================
run_script(
  "analysis/scripts/first_epoch_analysis.R", 
  "First epoch learning assessment"
)

# ===================================================
# SCRIPT 3: WITHIN-MODEL ITEM GROUP ANALYSIS (NULL)
# ===================================================
run_script(
  "analysis/scripts/within_model_itemgroup_posthoc.R",
  "Within-model item group analysis (null preference)"
)

# ===================================================
# SCRIPT 4: WITHIN-MODEL ITEM GROUP ANALYSIS (OVERT)
# ===================================================
run_script(
  "analysis/scripts/within_model_itemgroup_posthoc_overt.R",
  "Within-model item group analysis (overt preference)"
)

# ===================================================
# SCRIPT 5: END-STATE OVERT VS CHANCE (DETAILED)
# ===================================================
run_script(
  "analysis/scripts/endstate_overt_vs_chance_detailed.R",
  "End-state overt preference vs chance (comprehensive)"
)

# ===================================================
# SCRIPT 6: COMPREHENSIVE PAIRWISE COMPARISONS
# ===================================================
run_script(
  "analysis/scripts/overt_preference_pairwise_comparisons.R",
  "Comprehensive pairwise comparisons (Parts B & C)"
)

# ===================================================
# ANALYSIS PIPELINE COMPLETE
# ===================================================

end_time <- Sys.time()
total_runtime <- round(as.numeric(difftime(end_time, start_time, units = "mins")), 2)

cat("==========================================================\n")
cat("ANALYSIS PIPELINE COMPLETED\n")
cat("==========================================================\n")
cat("Started:  ", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n")
cat("Finished: ", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n")
cat("Total runtime:", total_runtime, "minutes\n\n")

# ===================================================
# SUMMARY OF GENERATED FILES
# ===================================================

cat("GENERATED FILES SUMMARY:\n")
cat("------------------------\n\n")

# Check for key output files
output_files <- c(
  # Main analysis outputs
  "analysis/tables/inflection_points_all.csv",
  "analysis/tables/inflection_points_successful.csv", 
  "analysis/tables/inflection_anova_results.csv",
  "analysis/tables/phase1_acquisition_contrasts.csv",
  "analysis/tables/phase2_form_sensitivity.csv",
  "analysis/tables/phase2_itemgroup_variance.csv",
  
  # First epoch
  "analysis/tables/first_epoch_checkpoints.csv",
  "analysis/tables/first_epoch_summary.csv",
  
  # Within-model analyses
  "analysis/tables/within_model_itemgroup_emmeans.csv",
  "analysis/tables/within_model_itemgroup_pairwise.csv",
  "analysis/tables/within_model_itemgroup_emmeans_overt.csv",
  "analysis/tables/within_model_itemgroup_pairwise_overt.csv",
  
  # Overt preference vs chance
  "analysis/tables/endstate_overt_vs_chance_models.csv",
  "analysis/tables/endstate_overt_vs_chance_itemgroups.csv",
  "analysis/tables/endstate_overt_vs_chance_forms.csv",
  "analysis/tables/endstate_overt_itemgroup_consistency.csv",
  
  # Pairwise comparisons
  "analysis/tables/pairwise_b1_number_contrasts.csv",
  "analysis/tables/pairwise_b2_person_contrasts.csv",
  "analysis/tables/pairwise_b3_control_contrasts.csv",
  "analysis/tables/pairwise_b4_expletive_contrasts.csv",
  "analysis/tables/pairwise_b5_topic_shift_contrasts.csv",
  "analysis/tables/pairwise_c1_forms_vs_default.csv",
  "analysis/tables/pairwise_c2_complex_embedding.csv",
  "analysis/tables/pairwise_c3_negation_types.csv"
)

# Report file status
files_found <- 0
files_missing <- 0

for (file in output_files) {
  if (file.exists(file)) {
    files_found <- files_found + 1
    file_size <- file.info(file)$size
    cat("✓", basename(file), "\n")
  } else {
    files_missing <- files_missing + 1
    cat("✗", basename(file), "(MISSING)\n")
  }
}

cat("\nFILE STATUS:\n")
cat("Files generated:", files_found, "/", length(output_files), "\n")
if (files_missing > 0) {
  cat("Files missing:", files_missing, "\n")
  cat("Check script outputs above for errors.\n")
}

# ===================================================
# REPORTS AVAILABLE
# ===================================================

cat("\nCOMPREHENSIVE REPORTS:\n")
cat("----------------------\n")

report_files <- c(
  "analysis/STATISTICAL_RESULTS_SUMMARY.md",
  "analysis/COMPREHENSIVE_APA_REPORT.md", 
  "analysis/PURE_STATISTICAL_REPORT.md",
  "analysis/COMPREHENSIVE_OVERT_PREFERENCE_REPORT.md",
  "analysis/STATISTICAL_TESTS_DOCUMENTATION.md",
  "analysis/BASELINE_RESULTS_TEMPLATE.md"
)

for (report in report_files) {
  if (file.exists(report)) {
    cat("✓", basename(report), "\n")
  } else {
    cat("✗", basename(report), "(not available)\n")
  }
}

cat("\n==========================================================\n")
cat("ANALYSIS COMPLETE! Check outputs above for any errors.\n")
cat("==========================================================\n")