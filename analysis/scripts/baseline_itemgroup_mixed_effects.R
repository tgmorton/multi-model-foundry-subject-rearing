# Baseline Model Item Group Mixed-Effects Pairwise Comparisons
# Using consistent methodology with odds ratios
# ===========================================================

library(tidyverse)
library(lme4)
library(emmeans)

# Load data
dat <- read_csv("evaluation/results/all_models_null_subject_lme4_ready.csv")

# Get baseline end-state data
baseline_data <- dat %>%
  filter(model == "exp0_baseline") %>%
  group_by(model) %>%
  filter(checkpoint_num >= max(checkpoint_num) - 1000) %>%
  ungroup() %>%
  filter(form_type == "overt")

# All pairwise comparisons for baseline model
all_results <- tibble()

# 1. PERSON CONTRASTS
cat("=== PERSON CONTRASTS ===\n")
person_data <- baseline_data %>%
  mutate(person = case_when(
    str_detect(item_group, "^1[ab]_") ~ "1st Person",
    str_detect(item_group, "^2[ab]_") ~ "2nd Person", 
    str_detect(item_group, "^3[ab]_") ~ "3rd Person",
    TRUE ~ "Other"
  )) %>%
  filter(person != "Other")

# 1st vs 2nd person
first_second <- person_data %>% filter(person %in% c("1st Person", "2nd Person"))
mod <- glmer(correct ~ person + (1|item_id), data = first_second, family = binomial)
emm <- emmeans(mod, ~ person)
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("1st vs 2nd Person: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")

# 1st vs 3rd person  
first_third <- person_data %>% filter(person %in% c("1st Person", "3rd Person"))
mod <- glmer(correct ~ person + (1|item_id), data = first_third, family = binomial)
emm <- emmeans(mod, ~ person)
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("1st vs 3rd Person: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")

# 2nd vs 3rd person
second_third <- person_data %>% filter(person %in% c("2nd Person", "3rd Person"))
mod <- glmer(correct ~ person + (1|item_id), data = second_third, family = binomial)
emm <- emmeans(mod, ~ person)
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("2nd vs 3rd Person: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")

# 2. NUMBER CONTRASTS  
cat("\n=== NUMBER CONTRASTS ===\n")

# 2nd person: singular vs plural
second_data <- baseline_data %>% 
  filter(str_detect(item_group, "^2[ab]_")) %>%
  mutate(number = ifelse(str_detect(item_group, "2a_"), "Singular", "Plural"))

mod <- glmer(correct ~ number + (1|item_id), data = second_data, family = binomial)
emm <- emmeans(mod, ~ number)
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("2nd Person Sing vs Plural: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")

# 1st person: singular vs plural
first_data <- baseline_data %>% 
  filter(str_detect(item_group, "^3[ab]_")) %>%
  mutate(number = ifelse(str_detect(item_group, "3a_"), "Singular", "Plural"))

mod <- glmer(correct ~ number + (1|item_id), data = first_data, family = binomial)
emm <- emmeans(mod, ~ number)
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("1st Person Sing vs Plural: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")

# 3. CONTROL CONTRASTS
cat("\n=== CONTROL CONTRASTS ===\n")
control_data <- baseline_data %>%
  filter(str_detect(item_group, "^4[ab]_")) %>%
  mutate(control_type = ifelse(str_detect(item_group, "4a_"), "Subject Control", "Object Control"))

mod <- glmer(correct ~ control_type + (1|item_id), data = control_data, family = binomial)
emm <- emmeans(mod, ~ control_type)
emm_summary <- summary(emm, type = "response")
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("Subject vs Object Control: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")
cat("Subject Control preference:", round(emm_summary$prob[1], 3), "\n")
cat("Object Control preference:", round(emm_summary$prob[2], 3), "\n")

# 4. EXPLETIVE CONTRASTS
cat("\n=== EXPLETIVE CONTRASTS ===\n")
expletive_data <- baseline_data %>%
  filter(str_detect(item_group, "^5[ab]_")) %>%
  mutate(expletive_type = ifelse(str_detect(item_group, "5a_"), "Seems", "Be"))

mod <- glmer(correct ~ expletive_type + (1|item_id), data = expletive_data, family = binomial)
emm <- emmeans(mod, ~ expletive_type)
emm_summary <- summary(emm, type = "response")
contrast <- pairs(emm, type = "response")
contrast_summary <- summary(contrast, infer = TRUE)
cat("Seems vs Be: OR =", round(contrast_summary$odds.ratio, 3), 
    ", CI = [", round(contrast_summary$asymp.LCL, 3), ",", round(contrast_summary$asymp.UCL, 3), 
    "], p =", round(contrast_summary$p.value, 3), "\n")
cat("Seems preference:", round(emm_summary$prob[1], 3), "\n")
cat("Be preference:", round(emm_summary$prob[2], 3), "\n")

cat("\nDone! Use these odds ratios for consistent reporting.\n")