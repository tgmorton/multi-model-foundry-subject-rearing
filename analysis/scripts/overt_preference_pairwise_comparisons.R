# Comprehensive Overt Preference Pairwise Comparisons
# PART B: Item Group Contrasts + PART C: Form/Processing Comparisons
# =================================================================

library(tidyverse)
library(lme4)
library(emmeans)
library(broom)

# Load the data
cat("Loading data...\n")
dat <- read_csv("evaluation/results/all_models_null_subject_lme4_ready.csv")

# Model labels
model_labels <- tribble(
  ~model, ~Model,
  "exp0_baseline", "Baseline",
  "exp1_remove_expletives", "Remove Expletives", 
  "exp2_impoverish_determiners", "Impoverish Determiners",
  "exp3_remove_articles", "Remove Articles",
  "exp4_lemmatize_verbs", "Lemmatize Verbs",
  "exp5_remove_subject_pronominals", "Remove Subject Pronominals"
)

dat <- dat %>%
  left_join(model_labels, by = "model")

# Get end-state data (last 1000 checkpoints)
endstate_data <- dat %>%
  group_by(model) %>%
  filter(checkpoint_num >= max(checkpoint_num) - 1000) %>%
  ungroup() %>%
  filter(form_type == "overt")  # Focus on overt preferences

cat("\n===========================================================\n")
cat("COMPREHENSIVE OVERT PREFERENCE PAIRWISE COMPARISONS\n")
cat("===========================================================\n")

# Store all results
all_results <- list()

# =================================================================
# PART B: PAIRWISE ITEM GROUP COMPARISONS
# =================================================================

cat("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
cat("PART B: PAIRWISE ITEM GROUP COMPARISONS\n")  
cat("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

# B1: NUMBER CONTRASTS (1a vs 1b, 2a vs 2b, 3a vs 3b)
cat("\n=== B1: NUMBER CONTRASTS ===\n")

number_pairs <- list(
  "3rd Person" = c("1a_3rdSG", "1b_3rdPL"),
  "2nd Person" = c("2a_2ndSG", "2b_2ndPL"), 
  "1st Person" = c("3a_1stSg", "3b_1stPL")
)

b1_results <- list()
b1_summary <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  model_summary <- list()
  
  for (pair_name in names(number_pairs)) {
    sg_item <- number_pairs[[pair_name]][1]
    pl_item <- number_pairs[[pair_name]][2]
    
    # Get data for both items
    sg_data <- model_data %>% filter(item_group == sg_item)
    pl_data <- model_data %>% filter(item_group == pl_item)
    
    if (nrow(sg_data) > 0 && nrow(pl_data) > 0) {
      # Calculate proportions
      sg_prop <- mean(sg_data$correct)
      pl_prop <- mean(pl_data$correct)
      
      # Contingency table for chi-square test
      cont_table <- matrix(
        c(sum(sg_data$correct), nrow(sg_data) - sum(sg_data$correct),
          sum(pl_data$correct), nrow(pl_data) - sum(pl_data$correct)),
        nrow = 2,
        byrow = TRUE,
        dimnames = list(c("Singular", "Plural"), c("Overt", "Null"))
      )
      
      # Chi-square test
      chi_test <- chisq.test(cont_table)
      
      # Store summary for table
      model_summary[[pair_name]] <- tibble(
        comparison = pair_name,
        sg_prop = sg_prop,
        pl_prop = pl_prop,
        diff = sg_prop - pl_prop,
        chi_sq = chi_test$statistic,
        p_value = chi_test$p.value,
        significant = chi_test$p.value < 0.05
      )
      
      # Store detailed result
      b1_results[[paste0(model_name, "_", gsub(" ", "_", pair_name))]] <- tibble(
        model = model_name,
        Model = model_label,
        comparison = pair_name,
        item_sg = sg_item,
        item_pl = pl_item,
        prop_sg = sg_prop,
        prop_pl = pl_prop,
        diff = sg_prop - pl_prop,
        chi_sq = chi_test$statistic,
        p_value = chi_test$p.value,
        significant = chi_test$p.value < 0.05
      )
    }
  }
  
  # Store model summary
  b1_summary[[model_name]] <- bind_rows(model_summary) %>%
    mutate(model = model_name, Model = model_label, .before = 1)
}

# Display summary table
cat("\nNUMBER CONTRASTS SUMMARY:\n")
cat("=========================\n")
all_b1_summary <- bind_rows(b1_summary)
for (model_name in unique(endstate_data$model)) {
  model_label <- all_b1_summary %>% filter(model == model_name) %>% pull(Model) %>% unique()
  model_data <- all_b1_summary %>% filter(model == model_name)
  
  cat(sprintf("\n%s:\n", model_label))
  for (i in 1:nrow(model_data)) {
    row <- model_data[i,]
    cat(sprintf("  %s: SG=%.3f, PL=%.3f, diff=%+.3f, χ²(1)=%.2f, p=%.4f %s\n",
                str_pad(row$comparison, 12), row$sg_prop, row$pl_prop, row$diff,
                row$chi_sq, row$p_value, ifelse(row$significant, "*", "")))
  }
}

# B2: PERSON CONTRASTS (1st vs 2nd vs 3rd)
cat("\n\n=== B2: PERSON CONTRASTS ===\n")

person_groups <- list(
  "1st Person" = c("3a_1stSg", "3b_1stPL"),
  "2nd Person" = c("2a_2ndSG", "2b_2ndPL"),
  "3rd Person" = c("1a_3rdSG", "1b_3rdPL")
)

b2_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\n%s:\n", model_label))
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # Calculate mean overt preference for each person
  person_means <- map_dfr(person_groups, function(items) {
    person_data <- model_data %>% filter(item_group %in% items)
    tibble(
      overt_prop = mean(person_data$correct),
      n_obs = nrow(person_data)
    )
  }, .id = "person")
  
  # Pairwise comparisons
  person_pairs <- combn(names(person_groups), 2, simplify = FALSE)
  
  for (pair in person_pairs) {
    p1_items <- person_groups[[pair[1]]]
    p2_items <- person_groups[[pair[2]]]
    
    p1_data <- model_data %>% filter(item_group %in% p1_items)
    p2_data <- model_data %>% filter(item_group %in% p2_items)
    
    if (nrow(p1_data) > 0 && nrow(p2_data) > 0) {
      p1_prop <- mean(p1_data$correct)
      p2_prop <- mean(p2_data$correct)
      
      # Chi-square test
      cont_table <- matrix(
        c(sum(p1_data$correct), nrow(p1_data) - sum(p1_data$correct),
          sum(p2_data$correct), nrow(p2_data) - sum(p2_data$correct)),
        nrow = 2, byrow = TRUE
      )
      
      chi_test <- chisq.test(cont_table)
      
      cat(sprintf("  %s vs %s: %.3f vs %.3f, diff=%.3f, p=%.4f %s\n",
                  pair[1], pair[2], p1_prop, p2_prop, p1_prop - p2_prop,
                  chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
      
      # Store result
      b2_results[[paste0(model_name, "_", gsub(" ", "_", pair[1]), "_vs_", gsub(" ", "_", pair[2]))]] <- tibble(
        model = model_name,
        Model = model_label,
        person1 = pair[1],
        person2 = pair[2],
        prop1 = p1_prop,
        prop2 = p2_prop,
        diff = p1_prop - p2_prop,
        chi_sq = chi_test$statistic,
        p_value = chi_test$p.value,
        significant = chi_test$p.value < 0.05
      )
    }
  }
}

# B3: CONTROL CONTRASTS (4a vs 4b)
cat("\n\n=== B3: CONTROL CONTRASTS ===\n")

b3_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  subj_data <- model_data %>% filter(item_group == "4a_subject_control")
  obj_data <- model_data %>% filter(item_group == "4b_object_control")
  
  if (nrow(subj_data) > 0 && nrow(obj_data) > 0) {
    subj_prop <- mean(subj_data$correct)
    obj_prop <- mean(obj_data$correct)
    
    # Chi-square test
    cont_table <- matrix(
      c(sum(subj_data$correct), nrow(subj_data) - sum(subj_data$correct),
        sum(obj_data$correct), nrow(obj_data) - sum(obj_data$correct)),
      nrow = 2, byrow = TRUE
    )
    
    chi_test <- chisq.test(cont_table)
    
    cat(sprintf("%s: Subject=%.3f, Object=%.3f, diff=%.3f, p=%.4f %s\n",
                str_pad(model_label, 25), subj_prop, obj_prop, subj_prop - obj_prop,
                chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
    
    # Store result
    b3_results[[model_name]] <- tibble(
      model = model_name,
      Model = model_label,
      subject_control = subj_prop,
      object_control = obj_prop,
      diff = subj_prop - obj_prop,
      chi_sq = chi_test$statistic,
      p_value = chi_test$p.value,
      significant = chi_test$p.value < 0.05
    )
  }
}

# B4: EXPLETIVE CONTRASTS (5a vs 5b)
cat("\n\n=== B4: EXPLETIVE CONTRASTS ===\n")

b4_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  seems_data <- model_data %>% filter(item_group == "5a_expletive_seems")
  be_data <- model_data %>% filter(item_group == "5b_expletive_be")
  
  if (nrow(seems_data) > 0 && nrow(be_data) > 0) {
    seems_prop <- mean(seems_data$correct)
    be_prop <- mean(be_data$correct)
    
    # Chi-square test
    cont_table <- matrix(
      c(sum(seems_data$correct), nrow(seems_data) - sum(seems_data$correct),
        sum(be_data$correct), nrow(be_data) - sum(be_data$correct)),
      nrow = 2, byrow = TRUE
    )
    
    chi_test <- chisq.test(cont_table)
    
    cat(sprintf("%s: Seems=%.3f, Be=%.3f, diff=%.3f, p=%.4f %s\n",
                str_pad(model_label, 25), seems_prop, be_prop, seems_prop - be_prop,
                chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
    
    # Store result
    b4_results[[model_name]] <- tibble(
      model = model_name,
      Model = model_label,
      expletive_seems = seems_prop,
      expletive_be = be_prop,
      diff = seems_prop - be_prop,
      chi_sq = chi_test$statistic,
      p_value = chi_test$p.value,
      significant = chi_test$p.value < 0.05
    )
  }
}

# B5: TOPIC SHIFT CONTRASTS (7a vs 7b)
cat("\n\n=== B5: TOPIC SHIFT CONTRASTS ===\n")

b5_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  no_shift_data <- model_data %>% filter(item_group == "7a_conjunction_no_topic_shift")
  shift_data <- model_data %>% filter(item_group == "7b_conjunction_topic_shift")
  
  if (nrow(no_shift_data) > 0 && nrow(shift_data) > 0) {
    no_shift_prop <- mean(no_shift_data$correct)
    shift_prop <- mean(shift_data$correct)
    
    # Chi-square test
    cont_table <- matrix(
      c(sum(no_shift_data$correct), nrow(no_shift_data) - sum(no_shift_data$correct),
        sum(shift_data$correct), nrow(shift_data) - sum(shift_data$correct)),
      nrow = 2, byrow = TRUE
    )
    
    chi_test <- chisq.test(cont_table)
    
    cat(sprintf("%s: No_shift=%.3f, Shift=%.3f, diff=%.3f, p=%.4f %s\n",
                str_pad(model_label, 25), no_shift_prop, shift_prop, no_shift_prop - shift_prop,
                chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
    
    # Store result
    b5_results[[model_name]] <- tibble(
      model = model_name,
      Model = model_label,
      no_topic_shift = no_shift_prop,
      topic_shift = shift_prop,
      diff = no_shift_prop - shift_prop,
      chi_sq = chi_test$statistic,
      p_value = chi_test$p.value,
      significant = chi_test$p.value < 0.05
    )
  }
}

# =================================================================
# PART C: FORM/PROCESSING COMPARISONS
# =================================================================

cat("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
cat("PART C: FORM/PROCESSING COMPARISONS\n")
cat("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

# C1: ALL FORMS VS DEFAULT BASELINE
cat("\n=== C1: ALL FORMS VS DEFAULT ===\n")

c1_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\n%s:\n", model_label))
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # Get default baseline
  default_data <- model_data %>% filter(form == "default")
  default_prop <- mean(default_data$correct)
  
  # Compare each form to default
  other_forms <- setdiff(unique(model_data$form), "default")
  
  for (form_name in other_forms) {
    form_data <- model_data %>% filter(form == form_name)
    
    if (nrow(form_data) > 0) {
      form_prop <- mean(form_data$correct)
      
      # Chi-square test
      cont_table <- matrix(
        c(sum(default_data$correct), nrow(default_data) - sum(default_data$correct),
          sum(form_data$correct), nrow(form_data) - sum(form_data$correct)),
        nrow = 2, byrow = TRUE
      )
      
      chi_test <- chisq.test(cont_table)
      
      cat(sprintf("  %s vs default: %.3f vs %.3f, diff=%.3f, p=%.4f %s\n",
                  str_pad(form_name, 20), form_prop, default_prop, form_prop - default_prop,
                  chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
      
      # Store result
      c1_results[[paste0(model_name, "_", form_name)]] <- tibble(
        model = model_name,
        Model = model_label,
        form = form_name,
        form_prop = form_prop,
        default_prop = default_prop,
        diff = form_prop - default_prop,
        chi_sq = chi_test$statistic,
        p_value = chi_test$p.value,
        significant = chi_test$p.value < 0.05
      )
    }
  }
}

# C2: COMPLEX EMBEDDING COMPARISONS (long vs embedded)
cat("\n\n=== C2: COMPLEX EMBEDDING COMPARISONS ===\n")

c2_results <- list()

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  long_data <- model_data %>% filter(form == "complex_long")
  emb_data <- model_data %>% filter(form == "complex_emb")
  
  if (nrow(long_data) > 0 && nrow(emb_data) > 0) {
    long_prop <- mean(long_data$correct)
    emb_prop <- mean(emb_data$correct)
    
    # Chi-square test
    cont_table <- matrix(
      c(sum(long_data$correct), nrow(long_data) - sum(long_data$correct),
        sum(emb_data$correct), nrow(emb_data) - sum(emb_data$correct)),
      nrow = 2, byrow = TRUE
    )
    
    chi_test <- chisq.test(cont_table)
    
    cat(sprintf("%s: Long=%.3f, Embedded=%.3f, diff=%.3f, p=%.4f %s\n",
                str_pad(model_label, 25), long_prop, emb_prop, long_prop - emb_prop,
                chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
    
    # Store result
    c2_results[[model_name]] <- tibble(
      model = model_name,
      Model = model_label,
      complex_long = long_prop,
      complex_embedded = emb_prop,
      diff = long_prop - emb_prop,
      chi_sq = chi_test$statistic,
      p_value = chi_test$p.value,
      significant = chi_test$p.value < 0.05
    )
  }
}

# C3: NEGATION TYPE COMPARISONS
cat("\n\n=== C3: NEGATION TYPE COMPARISONS ===\n")

c3_results <- list()

negation_forms <- c("context_negation", "target_negation", "both_negation")

for (model_name in unique(endstate_data$model)) {
  model_label <- endstate_data %>% 
    filter(model == model_name) %>% 
    pull(Model) %>% 
    unique()
  
  cat(sprintf("\n%s:\n", model_label))
  
  model_data <- endstate_data %>% filter(model == model_name)
  
  # Get negation data
  neg_data <- map(negation_forms, function(form) {
    form_data <- model_data %>% filter(form == !!form)
    if (nrow(form_data) > 0) {
      list(prop = mean(form_data$correct), data = form_data)
    } else {
      NULL
    }
  })
  names(neg_data) <- negation_forms
  neg_data <- neg_data[!map_lgl(neg_data, is.null)]
  
  # Pairwise comparisons
  if (length(neg_data) >= 2) {
    neg_pairs <- combn(names(neg_data), 2, simplify = FALSE)
    
    for (pair in neg_pairs) {
      data1 <- neg_data[[pair[1]]]$data
      data2 <- neg_data[[pair[2]]]$data
      prop1 <- neg_data[[pair[1]]]$prop
      prop2 <- neg_data[[pair[2]]]$prop
      
      # Chi-square test
      cont_table <- matrix(
        c(sum(data1$correct), nrow(data1) - sum(data1$correct),
          sum(data2$correct), nrow(data2) - sum(data2$correct)),
        nrow = 2, byrow = TRUE
      )
      
      chi_test <- chisq.test(cont_table)
      
      cat(sprintf("  %s vs %s: %.3f vs %.3f, diff=%.3f, p=%.4f %s\n",
                  str_pad(pair[1], 18), pair[2], prop1, prop2, prop1 - prop2,
                  chi_test$p.value, ifelse(chi_test$p.value < 0.05, "*", "")))
      
      # Store result
      c3_results[[paste0(model_name, "_", pair[1], "_vs_", pair[2])]] <- tibble(
        model = model_name,
        Model = model_label,
        form1 = pair[1],
        form2 = pair[2],
        prop1 = prop1,
        prop2 = prop2,
        diff = prop1 - prop2,
        chi_sq = chi_test$statistic,
        p_value = chi_test$p.value,
        significant = chi_test$p.value < 0.05
      )
    }
  }
}

# =================================================================
# SAVE ALL RESULTS
# =================================================================

cat("\n\n=== SAVING RESULTS ===\n")

# Combine all results
write_csv(bind_rows(b1_results), "analysis/tables/pairwise_b1_number_contrasts.csv")
write_csv(bind_rows(b2_results), "analysis/tables/pairwise_b2_person_contrasts.csv") 
write_csv(bind_rows(b3_results), "analysis/tables/pairwise_b3_control_contrasts.csv")
write_csv(bind_rows(b4_results), "analysis/tables/pairwise_b4_expletive_contrasts.csv")
write_csv(bind_rows(b5_results), "analysis/tables/pairwise_b5_topic_shift_contrasts.csv")
write_csv(bind_rows(c1_results), "analysis/tables/pairwise_c1_forms_vs_default.csv")
write_csv(bind_rows(c2_results), "analysis/tables/pairwise_c2_complex_embedding.csv")
write_csv(bind_rows(c3_results), "analysis/tables/pairwise_c3_negation_types.csv")

cat("\nPairwise comparison analysis complete!\n")
cat("Results saved to analysis/tables/pairwise_*.csv\n\n")
cat("* p < 0.05\n")