Null Subject Acquisition Analysis Plan

  Phase 1: Primary Analyses

  Analysis 1A: End-of-Training Success (Mixed Effects)

  Research Question: Did models successfully acquire overt preference?

  Model:
  Preference ~ Model * Form_Type + (1|Item) + (1|Form)

  Details:
  - DV: Final checkpoint preference (continuous, 0-1)
  - Fixed Effects:
    - Model (6 levels: Baseline, Remove Expletives, etc.)
    - Form_Type (2 levels: null, overt)
    - Model × Form_Type interaction
  - Random Effects: Item, Form
  - Key Tests:
    - Form_Type main effect: overt > null (p < .05)
    - Model × Form_Type: do some models fail to learn?
  - Effect Sizes: Cohen's d for Form_Type effect within each model

  ---
  Analysis 1B: Overall Acquisition Timing (ANOVA)

  Research Question: Do experimental manipulations affect acquisition speed?

  Model:
  Acquisition_Checkpoint ~ Model + Error(Item)

  Details:
  - DV: 50/50 crossover checkpoint (continuous)
  - IV: Model (6 levels)
  - Random: Item as error term if analyzing item-level data
  - Key Tests:
    - Model main effect (F-test)
    - Pairwise comparisons: Each manipulation vs Baseline
  - Effect Sizes: η² for model effect, Cohen's d for pairwise comparisons

  ---
  Phase 2: Granular Analyses

  Analysis 2A: Item-Level Acquisition Timing

  Research Question: Which grammatical contexts are most affected by each manipulation?

  Model:
  Item_Acquisition_Checkpoint ~ Model * Item_Group + (1|Item)

  Details:
  - DV: Item-specific 50/50 crossover checkpoint
  - Fixed Effects: Model, Item_Group, Model × Item_Group
  - Random: Item nested in Item_Group
  - Key Tests:
    - Model × Item_Group interactions
    - Simple effects: Model differences within each Item_Group
  - Follow-up: Which item groups show biggest manipulation effects?

  ---
  Analysis 2B: Form-Level Acquisition Timing

  Research Question: Which linguistic contexts are most affected by each manipulation?

  Model:
  Form_Acquisition_Checkpoint ~ Model * Form + (1|Item)

  Details:
  - DV: Form-specific 50/50 crossover checkpoint
  - Fixed Effects: Model, Form, Model × Form
  - Random: Item (forms tested across items)
  - Key Tests:
    - Model × Form interactions
    - Simple effects: Model differences within each Form
  - Follow-up: Rank forms by manipulation sensitivity

  ---
  Phase 3: Learning Dynamics

  Analysis 3A: Learning Trajectory Shapes

  Research Question: Do manipulations affect learning curve shape vs just timing?

  Model:
  Preference ~ Model * poly(Checkpoint, 3) + (1 + Checkpoint|Item) + (1|Form)

  Details:
  - DV: Preference at each checkpoint (full time series)
  - Fixed Effects: Model × polynomial time trends (linear, quadratic, cubic)
  - Random: Random intercepts and slopes for Item, random intercepts for Form
  - Key Tests: Model × polynomial interactions (shape differences)

  ---
  Analysis 3B: Acquisition Consistency

  Research Question: Are some manipulations more variable in their effects?

  Analysis:
  - DV: Standard deviation of acquisition times within each Model
  - Test: Levene's test for equality of variances
  - Visualization: Box plots of acquisition times by Model
  - Interpretation: Which manipulations create more/less consistent learning?

  ---
  Phase 4: Exploratory Analyses

  Analysis 4A: Threshold Sensitivity

  Research Question: Are results robust to different acquisition thresholds?

  Robustness Check:
  - Repeat Analysis 1B with 60/40, 70/30, 80/20 thresholds
  - Examine correlation between acquisition times across thresholds
  - Expected: Strong correlations indicate robust model rankings

  Analysis 4B: Early vs Late Acquisition

  Research Question: Do some manipulations affect early vs late learning differently?

  Approach:
  - Split acquisition times at median
  - DV: Early vs Late acquisition (binary)
  - Test: Logistic regression with Model predictor
  - Interpretation: Which manipulations cause delayed vs failed acquisition?

  ---
  Planned Contrasts & Effect Sizes

  Theoretically-Motivated Contrasts:

  1. Baseline vs All Manipulations (general impairment effect)
  2. Remove Expletives vs Remove Articles (expletive-specific effects)
  3. Lemmatize Verbs vs Others (morphological vs syntactic effects)
  4. Remove Subject Pronominals vs Others (subject-specific effects)

  Effect Size Reporting:

  - Cohen's d for pairwise comparisons (small: 0.2, medium: 0.5, large: 0.8)
  - η² for omnibus effects (small: 0.01, medium: 0.06, large: 0.14)
  - 95% Confidence Intervals for all effect sizes

  ---
  Multiple Comparisons Correction:

  - Phase 1: No correction (primary hypotheses)
  - Phase 2: FDR correction within each analysis
  - Phase 3-4: FDR correction (exploratory)
