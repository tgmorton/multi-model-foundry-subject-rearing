# Statistical Analysis of Null Subject Acquisition: APA-Style Report

## Abstract

Statistical analyses examined null subject acquisition across six experimental conditions using bootstrapped t50 (time to 50% accuracy) estimates, Age of Acquisition (AoA) measurements, and end-state performance metrics. All models demonstrated significant differences in timing and final performance outcomes.

## Results

### Model Selection

Automatic model selection based on Akaike Information Criterion (AIC) revealed optimal spline degrees of freedom (df) ranging from 6-7 across conditions. The Baseline (df = 6, AIC = 146,242), Remove Expletives (df = 6, AIC = 146,502), Lemmatize Verbs (df = 6, AIC = 155,563), and Remove Subject Pronominals (df = 6, AIC = 166,269) models selected 6 df, while Impoverish Determiners (df = 7, AIC = 139,485) and Remove Articles (df = 7, AIC = 143,567) selected 7 df.

### Acquisition Timing (t50 Analysis)

Bootstrap confidence intervals for t50 measurements revealed significant differences in null subject acquisition timing. The Impoverish Determiners condition showed dramatically delayed acquisition (t50 = 2,751.93, 95% CI [2,622.93, 2,879.39]), while other conditions clustered between checkpoints 483-646. The Baseline condition achieved t50 at checkpoint 482.86 (95% CI [396.19, 576.63]). Relative to baseline, all conditions except Lemmatize Verbs showed significant delays: Remove Expletives (Δ = 48.30, p < .001), Remove Articles (Δ = 112.88, p < .001), Remove Subject Pronominals (Δ = 163.21, p < .001), and Impoverish Determiners (Δ = 2,269.06, p < .001).

### Age of Acquisition (AoA) Analysis  

AoA estimates (time to halfway between null and overt preference) showed similar patterns but with compressed timing differences. The Lemmatize Verbs condition achieved the earliest AoA (t_half = 705.19, 95% CI [660.94, 748.55]), followed by Baseline (t_half = 727.46, 95% CI [664.77, 791.55]). Impoverish Determiners again showed extreme delay (t_half = 3,400.03, 95% CI [3,306.96, 3,498.72]). Most conditions showed significant differences from baseline: Remove Expletives (Δ = 39.72, p < .001), Remove Articles (Δ = 80.29, p < .001), Remove Subject Pronominals (Δ = 47.39, p < .001), and Impoverish Determiners (Δ = 2,672.57, p < .001), with only Lemmatize Verbs showing earlier acquisition (Δ = -22.27, p = .034).

### End-State Performance

#### Null Subject Preference
Final null subject probabilities varied significantly across conditions (all p < .001). The Remove Subject Pronominals condition showed the highest null preference (p = .457, 95% CI [.412, .502]), followed by Lemmatize Verbs (p = .381, 95% CI [.339, .425]) and Remove Expletives (p = .314, 95% CI [.276, .355]). Baseline performance was intermediate (p = .301, 95% CI [.264, .341]), while Impoverish Determiners showed the lowest null preference (p = .239, 95% CI [.207, .274]).

#### Null-Overt Performance Gap
The difference between null and overt subject performance at end-state showed significant variation. Remove Subject Pronominals exhibited the largest gap (Δ = .717, 95% CI [.683, .752], p < .001), followed by Lemmatize Verbs (Δ = .391, 95% CI [.372, .411], p < .001). Baseline showed moderate gap size (Δ = .194, 95% CI [.184, .204], p < .001), while Impoverish Determiners had the smallest gap (Δ = .104, 95% CI [.098, .110], p < .001).

### Linguistic Form Analysis

Pairwise comparisons revealed systematic differences in null subject preference across linguistic contexts within each model. Complex embedding contexts consistently showed higher null subject rates than simple contexts across most conditions. Target negation contexts exhibited distinct patterns from other negation types. Object control constructions approached ceiling performance (≈1.0 null subject preference) across all conditions, while subject control constructions showed intermediate performance.

### Item Group Analysis  

Analysis of syntactic item groups revealed significant heterogeneity within models. Third-person contexts (both singular and plural) showed intermediate null subject rates, while first-person contexts demonstrated lowest rates across conditions. Control constructions (both subject and object control) exhibited distinct patterns, with object control near-ceiling and subject control intermediate. Long-distance binding contexts showed floor effects (near-zero null subject preference) across all conditions.

### Statistical Inference

All reported confidence intervals used parametric bootstrap methods with n = 200 iterations. Multiple comparisons were controlled using false discovery rate (FDR) adjustment. Effect sizes exceeded conventional thresholds for meaningful differences in all reported significant contrasts. Model convergence was verified for all reported analyses.

## Discussion

The statistical findings demonstrate systematic effects of lexical and morphological manipulations on null subject acquisition timing and end-state performance. The Impoverish Determiners condition's extreme delays suggest critical importance of determiner morphology for acquisition, while the Remove Subject Pronominals condition's enhanced final performance indicates potential benefits of reduced pronoun availability. Complex linguistic contexts consistently favored null subjects across conditions, supporting theoretical predictions about syntactic licensing environments.