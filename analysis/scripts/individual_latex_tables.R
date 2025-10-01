# Individual LaTeX Table Generator - Using Corrected Mixed-Effects Results
# ========================================================================

library(tidyverse)
library(xtable)

# Create directories
dir.create("analysis/tables/latex_tables", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables/latex_tables/itemgroups", recursive = TRUE, showWarnings = FALSE)
dir.create("analysis/tables/latex_tables/forms", recursive = TRUE, showWarnings = FALSE)

# Load corrected mixed-effects results
comprehensive_data <- read_csv("analysis/tables/final_comprehensive_mixed_effects_comparisons.csv", show_col_types = FALSE)

# Model labels for clean names
model_labels <- tribble(
  ~Model, ~model_clean_name,
  "Baseline", "exp0baseline",
  "Remove Expletives", "exp1removeexpletives", 
  "Impoverish Determiners", "exp2impoverishdeterminers",
  "Remove Articles", "exp3removearticles",
  "Lemmatize Verbs", "exp4lemmatizeverbs",
  "Remove Subject Pronominals", "exp5removesubjectpronominals"
)

# Function to clean up item group names for display
clean_item_group_name <- function(item_name) {
  # Remove "Group " prefix and clean up underscores
  cleaned <- str_replace_all(item_name, "_", "\\_")
  return(cleaned)
}

# Function to order item groups properly 
order_item_groups <- function(data) {
  data %>%
    mutate(
      sort_order = case_when(
        str_detect(Comparison, "1st.*Singular.*Plural") ~ 1,
        str_detect(Comparison, "2nd.*Singular.*Plural") ~ 2,
        str_detect(Comparison, "3rd.*Singular.*Plural") ~ 3,
        str_detect(Comparison, "1st.*2nd") ~ 4,
        str_detect(Comparison, "1st.*3rd") ~ 5,
        str_detect(Comparison, "2nd.*3rd") ~ 6,
        str_detect(Comparison, "Subject.*Object") ~ 7,
        str_detect(Comparison, "Seems.*Be") ~ 8,
        str_detect(Comparison, "Topic.*Shift") ~ 9,
        TRUE ~ 99
      )
    ) %>%
    arrange(sort_order) %>%
    select(-sort_order)
}

# Generate tables for each model
for (model_name in unique(model_labels$Model)) {
  model_clean_name <- model_labels %>% 
    filter(Model == model_name) %>% 
    pull(model_clean_name)
  
  # ===================
  # ITEM GROUPS TABLE
  # ===================
  
  # Get item group pairwise comparisons for this model
  itemgroup_results <- comprehensive_data %>%
    filter(Model == model_name, 
           `Analysis Category` == "Item Group Pairwise") %>%
    order_item_groups() %>%
    mutate(
      # Extract preference percentages from Group 1 and Group 2
      group1_pct = as.numeric(str_remove(`Group 1 %`, "%")),
      group2_pct = as.numeric(str_remove(`Group 2 %`, "%")),
      # Calculate weighted average preference for each comparison
      avg_pref = (group1_pct + group2_pct) / 2 / 100,
      # Clean up comparison names
      comparison_clean = str_replace_all(Comparison, "_", "\\_")
    )
  
  # Format for LaTeX - Item Groups
  latex_itemgroups <- itemgroup_results %>%
    mutate(
      `Comparison` = comparison_clean,
      `Group 1` = clean_item_group_name(`Group 1`),
      `Group 1 %` = `Group 1 %`,
      `Group 2` = clean_item_group_name(`Group 2`),
      `Group 2 %` = `Group 2 %`,
      `OR` = `Odds Ratio`,
      `p-value` = `p-corrected`
    ) %>%
    select(`Comparison`, `Group 1`, `Group 1 %`, `Group 2`, `Group 2 %`, `OR`, `OR 95% CI`, `p-value`)
  
  # Generate LaTeX table
  xtab <- xtable(latex_itemgroups, 
                 caption = sprintf("%s: Item Group Pairwise Comparisons (Mixed-Effects + FDR Corrected)", model_name),
                 label = sprintf("tab:%s_itemgroups", model_clean_name),
                 align = c("l", "l", "l", "r", "l", "r", "r", "l", "l"))
  
  # Save to file
  itemgroup_file <- sprintf("analysis/tables/latex_tables/itemgroups/%s_itemgroups.tex", model_clean_name)
  cat(print(xtab, 
            include.rownames = FALSE,
            caption.placement = "top",
            table.placement = "",
            floating = FALSE,
            print.results = FALSE),
      file = itemgroup_file)
  
  cat(sprintf("Saved: %s\n", itemgroup_file))
  
  # =================
  # FORMS TABLE
  # =================
  
  # Get form pairwise comparisons for this model
  form_results <- comprehensive_data %>%
    filter(Model == model_name, 
           `Analysis Category` == "Form Pairwise") %>%
    # Order forms logically 
    mutate(
      sort_order = case_when(
        str_detect(Comparison, "Complex.*Long") ~ 1,
        str_detect(Comparison, "Target.*Context") ~ 2,
        str_detect(Comparison, "Target.*Both") ~ 3,
        TRUE ~ 99
      )
    ) %>%
    arrange(sort_order) %>%
    select(-sort_order)
  
  # Format for LaTeX - Forms
  latex_forms <- form_results %>%
    mutate(
      `Comparison` = str_replace_all(Comparison, "_", "\\_"),
      `Group 1` = clean_item_group_name(`Group 1`),
      `Group 1 %` = `Group 1 %`,
      `Group 2` = clean_item_group_name(`Group 2`), 
      `Group 2 %` = `Group 2 %`,
      `OR` = `Odds Ratio`,
      `p-value` = `p-corrected`
    ) %>%
    select(`Comparison`, `Group 1`, `Group 1 %`, `Group 2`, `Group 2 %`, `OR`, `OR 95% CI`, `p-value`)
  
  # Generate LaTeX table
  xtab <- xtable(latex_forms,
                 caption = sprintf("%s: Form Pairwise Comparisons (Mixed-Effects + FDR Corrected)", model_name),
                 label = sprintf("tab:%s_forms", model_clean_name),
                 align = c("l", "l", "l", "r", "l", "r", "r", "l", "l"))
  
  # Save to file
  forms_file <- sprintf("analysis/tables/latex_tables/forms/%s_forms.tex", model_clean_name)
  cat(print(xtab,
            include.rownames = FALSE,
            caption.placement = "top", 
            table.placement = "",
            floating = FALSE,
            print.results = FALSE),
      file = forms_file)
  
  cat(sprintf("Saved: %s\n", forms_file))
}

cat("\nAll corrected mixed-effects LaTeX tables saved!\n")
cat("\nDirectory structure:\n")
cat("analysis/tables/latex_tables/\n")
cat("├── itemgroups/\n")
cat("│   ├── exp0baseline_itemgroups.tex\n")
cat("│   ├── exp1removeexpletives_itemgroups.tex\n")
cat("│   ├── exp2impoverishdeterminers_itemgroups.tex\n")
cat("│   ├── exp3removearticles_itemgroups.tex\n")
cat("│   ├── exp4lemmatizeverbs_itemgroups.tex\n")
cat("│   └── exp5removesubjectpronominals_itemgroups.tex\n")
cat("└── forms/\n")
cat("    ├── exp0baseline_forms.tex\n")
cat("    ├── exp1removeexpletives_forms.tex\n")
cat("    ├── exp2impoverishdeterminers_forms.tex\n")
cat("    ├── exp3removearticles_forms.tex\n")
cat("    ├── exp4lemmatizeverbs_forms.tex\n")
cat("    └── exp5removesubjectpronominals_forms.tex\n")

cat("\n=== UPDATED TO USE CORRECTED MIXED-EFFECTS RESULTS ===\n")
cat("- All tables now use mixed-effects logistic regression with odds ratios\n")
cat("- FDR correction applied for multiple comparisons\n")
cat("- Consistent methodology across all analyses\n")
cat("\nTo include in your LaTeX document:\n")
cat("\\input{analysis/tables/latex_tables/itemgroups/exp0baseline_itemgroups.tex}\n")
cat("\\input{analysis/tables/latex_tables/forms/exp0baseline_forms.tex}\n")
cat("\n(Or use \\include{} if you prefer)\n")