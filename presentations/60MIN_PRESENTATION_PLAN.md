# 60-Minute Presentation Plan: Chapter 1 Full Proposal with Pilot Evidence

## Overview
**Total Time**: 60 minutes (50 min presentation + 10 min Q&A)
**Total Slides**: ~56 slides (averaging ~1 min per slide, with key slides taking longer)
**Format**: Project proposal showcasing theoretical motivation, comprehensive design, pilot results, and future work

---

## TIMING BREAKDOWN

### Part I: Theoretical Motivation (12 minutes) - Slides 1-22
### Part II: Proposed Research Design (8 minutes) - Slides 23-30  
### Part III: Pilot Evidence (20 minutes) - Slides 31-44
### Part IV: Remaining Work (6 minutes) - Slides 45-52
### Part V: Synthesis (4 minutes) - Slides 53-56
### Q&A: 10 minutes

---

## DETAILED SLIDE-BY-SLIDE BREAKDOWN WITH FIGURES

---

## PART I: THEORETICAL MOTIVATION (12 minutes)
**Keep from Oct22Meeting.tex - Acts 1-3**

### Slide 1: Title Slide (30 seconds)
**Content**: Title, name, date, affiliation
**Transition**: "Today I'll present a project that uses computational models to address a 40-year-old problem in language acquisition."

### Slide 2: Opening Statement (1 minute)
**Content**: Three core statements about investigating human learning, using LLMs as candidate models, null subjects as case study
**From**: Oct22Meeting.tex opening statements
**Transition**: "Let me start by explaining what null subjects are and why they're interesting."

### Slide 3: Cross-Linguistic Variation (1 minute)
**Content**: Italian vs. English null subject examples
**From**: Oct22Meeting.tex Slide 3-4
**Transition**: "This variation poses a fundamental learning problem."

### Slide 4: Plato's Problem (1.5 minutes)
**Content**: Poverty of stimulus, positive evidence only, how children learn without negative feedback
**From**: Oct22Meeting.tex Slide 5
**Key Point**: This is THE fundamental problem
**Transition**: "This is what Chomsky called the Poverty of Stimulus. For 60 years, linguists have been trying to explain how this learning happens."

### Slide 5: The Acquisition Challenge (1 minute)
**Content**: Empirical pattern - English children start at ~30% null, Italian at ~70%, no negative evidence
**From**: Oct22Meeting.tex Slide 6
**Transition**: "So linguists started proposing theories about what evidence children might use..."

### Slide 6: The Parametric Approach (1 minute)
**Content**: Binary parameter, UG, elegant solution to learnability
**From**: Oct22Meeting.tex Slide 7
**Transition**: "Rizzi and Chomsky proposed that multiple properties cluster together..."

### Slide 7: The Chomsky-Rizzi Cluster (1 minute)
**Content**: 4 properties that should cluster, deductive learning advantage
**From**: Oct22Meeting.tex Slide 8
**Transition**: "The beauty was that a child could detect ANY of these and deduce the others. But..."

### Slide 8: Cluster Properties Examples (45 seconds)
**Content**: Free inversion and that-trace examples
**From**: Oct22Meeting.tex Slide 9
**Skip if time pressure**: Can cut this for 60-min version

### Slide 9: Gilligan's Challenge (1 minute)
**Content**: Cluster doesn't hold across 100 languages
**From**: Oct22Meeting.tex Slide 14
**Transition**: "So researchers pivoted to different evidence types..."

### Slide 10: Hyams' Reversal (1 minute)
**Content**: AG/PRO parameter, children START with pro-drop, need triggers to reset
**From**: Oct22Meeting.tex Slide 11
**Transition**: "Hyams proposed two trigger types..."

### Slide 11: Trigger 1 - Expletives (45 seconds)
**Content**: It/there signal non-null-subject language
**From**: Oct22Meeting.tex Slide 12
**Transition**: "And second..."

### Slide 12: Trigger 2 - Overt Pronouns (45 seconds)
**Content**: Avoid Pronoun Principle, overt pronouns in neutral contexts
**From**: Oct22Meeting.tex Slide 13

### Slide 13: A Richer Typology (1 minute)
**Content**: CNSL, PNSL, SNSL, NNSL - 4 types, not binary
**From**: Oct22Meeting.tex Slide 15
**Transition**: "This led to different theories about morphology..."

### Slide 14: The Morphological Turn (1 minute)
**Content**: Jaeggli & Safir uniformity, Hyams reanalysis, problems (Finnish, Kriyol)
**From**: Oct22Meeting.tex Slide 17
**Skip if time pressure**: Can cut for 60-min version

### Slide 15: Alternative Perspectives (1 minute)
**Content**: Valian frequency, Wang null objects, Optional Infinitives
**From**: Oct22Meeting.tex Slide 18
**Skip if time pressure**: Can cut for 60-min version

### Slide 16: Modern Synthesis (1 minute)
**Content**: Features & economy, statistical learning, variational models
**From**: Oct22Meeting.tex Slide 19

### Slide 17: 40 Years of Proposed Evidence (1 minute)
**Content**: Table summarizing evidence types from all theories
**From**: Oct22Meeting.tex Slide 20
**Figure**: Table showing Abstract Syntax, Lexical Items, Morphology, Frequency, etc.
**Key Slide**: This shows the landscape of competing theories
**Transition**: "So we have all these competing theories, but here's the problem..."

### Slide 18: The Fundamental Problem (1.5 minutes)
**Content**: Can't test on children - unethical, impractical, confounded
**From**: Oct22Meeting.tex Slide 21
**Key Slide**: This is the impasse
**Transition**: "What we would need is ablation studies, isolation studies, controlled input. We can't do this with children, but..."

### Slide 19: What We Need (1 minute)
**Content**: Requirements - systematic isolation, ablation, precise control, transparent inspection, developmental tracking
**From**: Oct22Meeting.tex Slide 22
**Transition**: "This is where Small Language Models come in."

---

## PART II: PROPOSED RESEARCH DESIGN (8 minutes)
**NEW CONTENT - Comprehensive project description**

### Slide 20: SLMs as Experimental Systems (1.5 minutes)
**Content**: Key properties (learn from distributions, no explicit rules, controlable input, ablations possible, transparent, fast)
**From**: Oct22Meeting.tex Slide 23 EXPANDED
**Key Message**: "SLMs let us do the experiments we can't do on children"
**Transition**: "Here's what I'm proposing to do systematically..."

### Slide 21: Chapter 1 - Research Questions (1 minute)
**Content**: 
- What information contributes to learning subject drop?
- How much does direct vs. indirect evidence matter?
- What is the most important indirect evidence?
- How do statistical learners solve problems without negative evidence?
**From**: INTEGRATION_PLAN.md Slide 23
**Transition**: "To answer these, I'm using a controlled rearing approach..."

### Slide 22: Full Experimental Design Overview (1.5 minutes)
**Content**: Three study types:
1. Single ablations (6 ablations)
2. Continuous sweeps (pronoun ratio, determiner richness)
3. Combined ablations (all indirect, all direct+indirect, minimal)
**From**: INTEGRATION_PLAN.md Slide 25
**Figure**: Could create flowchart showing experiment types
**Key Slide**: Shows scope of full project
**Transition**: "Let me break down each component..."

### Slide 23: Cross-Linguistic & Cross-Architecture Design (1 minute)
**Content**: 
- Languages: English, Italian, counterfactuals
- Architectures: GPT-2 (S/M/L/XL), LSTM, BERT, n-grams
- Multiple runs per condition
**From**: INTEGRATION_PLAN.md Slide 26
**Transition**: "How will we evaluate these models?"

### Slide 24: Evaluation Framework (1 minute)
**Content**: 
- Behavioral: grammaticality, production, trajectories
- Representational: probing, similarity, transfer
- Statistical: AoA, effect sizes, learning curves
**From**: INTEGRATION_PLAN.md Slide 27
**Transition**: "One technical challenge is creating counterfactual languages..."

### Slide 25: Counterfactual Implementation (45 seconds)
**Content**: How to create "English with null subjects"
- BERT insertion
- Dummy pronouns
- Seq2seq
- LLM assistance
**From**: INTEGRATION_PLAN.md Slide 28
**Skip if time pressure**: Can cut for 60-min version

### Slide 26: Dataset Specifications (45 seconds)
**Content**: BabyLM 90M words, composition, eval sets with minimal pairs
**From**: INTEGRATION_PLAN.md Slide 29
**Transition**: "Now let me show you what I've already done as pilot work..."

### Slide 27: Timeline & Pre-Registration (1 minute)
**Content**: 4 phases, what's completed vs. planned, pre-registration plan
**From**: INTEGRATION_PLAN.md Slide 30
**Figure**: Could create Gantt chart or timeline graphic
**Key Message**: "Phase 1 is complete - that's what I'll show you next"
**Transition**: "So let me walk you through the pilot study..."

---

## PART III: PILOT EVIDENCE (20 minutes)
**From Presentation.tex - Chapter 1 Results**

### Slide 28: Pilot Study Overview (1 minute)
**Content**: What we've done - 5 ablations + baseline, English only, GPT-2 Small, BabyLM
**From**: INTEGRATION_PLAN.md Slide 31
**Key Message**: "This is proof of concept showing the approach works"
**Transition**: "Let me describe the materials..."

### Slide 29: Materials - BabyLM Dataset (1 minute)
**Content**: 90M corpus, child-directed speech weighted, composition breakdown
**From**: Presentation.tex frame "Materials: BabyLM Dataset"
**Figure Reference**: Could show corpus composition pie chart if you create one

### Slide 30: Evaluation Stimuli (1 minute)
**Content**: Minimal pairs design, core contrasts (Person/Number, Control, Expletives, Topic shift)
**From**: Presentation.tex frame "Evaluation Stimuli: Null vs. Overt Subjects"
**Transition**: "We also manipulated processing demands..."

### Slide 31: Processing Manipulations (45 seconds)
**Content**: Context complexity (simple, long NPs, embedded) and negation (target, context, both)
**From**: Presentation.tex frame "Processing Manipulations"
**Skip if time pressure**: Can mention briefly but cut detail

### Slide 32: Statistical Approach (1 minute)
**Content**: Binary outcome (overt preference), logistic GLMMs, AoA metrics
**From**: Presentation.tex frames "Measures and Analysis" + "Age of Acquisition"
**Figure**: Could show sample learning curve with annotations explaining AoA50
**Transition**: "Now let me show you what we found..."

### Slide 33: Experimental Design Summary (45 seconds)
**Content**: Quick recap of 6 conditions (0-5) mapping to theories
**From**: Presentation.tex frame "Experimental Design: Controlled Rearing"
**Figure**: Simple table showing:
```
0. Baseline (control)
1. Remove Expletives (Yang)
2. Impoverish Determiners (Duguine)  
3. Remove Articles (Duguine)
4. Lemmatize Verbs (Hyams)
5. Remove Pronouns (Hyams direct evidence)
```
**Transition**: "Let's start with the baseline..."

### Slide 34: Baseline Results (2 minutes)
**Content**: Developmental trajectory, 63.4% null initially, AoA=727, 69.6% overt finally
**From**: Presentation.tex frame "Baseline Model -- Training Curves"
**FIGURE**: `analysis/paper_figures/main/model_baseline.pdf`
**Key Finding**: All models start null, shift to overt - mirrors children
**Transition**: "This baseline shows the models CAN learn. Now let's see what happens when we remove evidence..."

### Slide 35: Exp 1 - Remove Expletives (1.5 minutes)
**Content**: AoA=767 (+39 epochs), no difference in endpoints
**From**: Presentation.tex frame "Exp 1: 'Remove Expletives'"
**FIGURE**: `analysis/paper_figures/wide/comparison_vs_baseline_overt_only_remove_expletives.pdf`
**Interpretation**: Expletives helpful but not critical - challenges Yang
**Transition**: "Expletives barely matter. But watch what happens with determiners..."

### Slide 36: Exp 2 - Impoverish Determiners (2 minutes)
**Content**: AoA=3400 (+2672 epochs!), strongest final preference, possible grokking
**From**: Presentation.tex frame "Exp 2: 'Impoverish Determiners'"
**FIGURE**: `analysis/paper_figures/wide/comparison_vs_baseline_overt_only_impoverish_determiners.pdf`
**Key Finding**: MASSIVE delay - determiners provide crucial shortcuts
**Transition**: "This is the biggest effect we see. Removing determiners entirely..."

### Slide 37: Exp 3 - Remove Articles (1.5 minutes)
**Content**: AoA=807 (+80 epochs), stronger null preference initially, lower overt finally
**From**: Presentation.tex frame "Exp 3: 'Remove Articles'"
**FIGURE**: `analysis/paper_figures/wide/comparison_vs_baseline_overt_only_remove_articles.pdf`
**Interpretation**: Articles matter ~2x more than expletives
**Transition**: "Now here's a surprise with morphology..."

### Slide 38: Exp 4 - Lemmatize Verbs (1.5 minutes)
**Content**: AoA=705 (-22 epochs, FASTER!), challenges Hyams
**From**: Presentation.tex frame "Exp 4: 'Lemmatize Verbs'"
**FIGURE**: `analysis/paper_figures/wide/comparison_vs_baseline_overt_only_lemmatize_verbs.pdf`
**Key Finding**: Morphology interferes, doesn't help - opposite of prediction
**Transition**: "So morphology actually slows learning. But the most critical evidence is..."

### Slide 39: Exp 5 - Remove Pronouns (2 minutes)
**Content**: AoA=774 (unstable), 54.4% overt (near chance), weakest learning
**From**: Presentation.tex frame "Exp 5: 'Remove Subject Pronominals'"
**FIGURE**: `analysis/paper_figures/wide/comparison_vs_baseline_overt_only_remove_subject_pronominals.pdf`
**Key Finding**: Direct evidence NECESSARY - without pronouns, barely learns
**Transition**: "Let me show you all six conditions together..."

### Slide 40: Cross-Model Comparison (1.5 minutes)
**Content**: All models overlaid showing rank order
**From**: Presentation.tex frame "Cross-Model Comparison"
**FIGURE**: `analysis/paper_figures/wide/all_models_comparison_log.pdf`
**Key Slide**: Shows evidence hierarchy visually
**Transition**: "So we see a clear ranking. But remember I also tested processing effects..."

### Slide 41: Processing Effects - Introduction (30 seconds)
**Content**: Bloom's prediction vs. our test
**From**: Presentation.tex frame "Processing Account: Predicted vs. Observed"
**Transition**: "Here's what we found..."

### Slide 42: Processing Effects - Results (1.5 minutes)
**Content**: Negation increases overt (opposite of prediction), complexity no effect
**From**: Presentation.tex frame "Exp 5: 'Remove Subject Pronominals' -- Training Curves" (forest plot)
**FIGURE**: `analysis/paper_figures/supplementary/forest_form_baseline.pdf`
**Key Finding**: Models behave OPPOSITE to processing account prediction
**Transition**: "This pattern is consistent across models..."

### Slide 43: Processing Effects Across Models (1 minute)
**Content**: Table showing which forms affect which models
**From**: Presentation.tex frame "Processing Effects Across All Models"
**FIGURE**: Table from `analysis/tables/latex_tables/forms_vs_default_checkmarks`
**Key Point**: Negation universally increases overt, complexity doesn't
**Skip if time pressure**: Can combine with previous slide

### Slide 44: Universal Null-First Stage (1.5 minutes)
**Content**: ALL models start null despite English being non-null-subject
**From**: Presentation.tex frame "Universal Early Null Subject Stage"
**Key Question**: Architecture bias or environmental evidence?
**Transition**: "This finding has theoretical implications..."

### Slide 45: Evidence Hierarchy (1.5 minutes)
**Content**: Rank order - Pronouns critical, Determiners shortcuts, Expletives minimal, Morphology interfering
**From**: Presentation.tex frame "Evidence Types: Shortcuts vs. Deep Learning"
**FIGURE**: Could show flowchart or ranking:
```
CRITICAL: Pronouns (54% without them)
HELPFUL: Determiners (+2672 epochs without)
HELPFUL: Articles (+80 epochs without)
MINIMAL: Expletives (+39 epochs without)
INTERFERING: Morphology (-22 epochs without)
```
**Transition**: "What does this mean for acquisition theory?"

### Slide 46: Theoretical Implications (1.5 minutes)
**Content**: What pilot challenges vs. supports
**From**: Presentation.tex frame "Broader Theoretical Implications"
**Key Message**: 
- Challenges: Yang's expletives, simple parameters, morphology triggers, processing accounts
- Supports: Hyams direct evidence, Duguine determiners, gradual learning
**Transition**: "This pilot proves the approach works. Now here's what remains..."

---

## PART IV: REMAINING WORK (6 minutes)
**From Research Plan Document**

### Slide 47: What Remains - Single Ablations (1 minute)
**Content**: Completed (✓) vs. Still Needed (□)
**From**: INTEGRATION_PLAN.md Slide 45
**Figure**: Checklist showing:
```
COMPLETED (English):
✓ Determiners
✓ Morphology  
✓ Pronouns
✓ Expletives

REMAINING:
□ Tense/aspect marking
□ Subject drop insertion
□ Italian replications
□ Counterfactuals
```

### Slide 48: What Remains - Continuous Sweeps (1.5 minutes)
**Content**: Gradients of evidence strength, pronoun:null ratio, determiner richness
**From**: INTEGRATION_PLAN.md Slide 46
**Figure**: Could show conceptual diagram of sweeps:
```
Pronoun:Null ratio: 100:0 → 70:30 → 50:50 → 30:70 → 0:100
Expected: Sigmoidal learning curve with threshold
```
**Transition**: "We'll also test combinations..."

### Slide 49: What Remains - Combined Ablations (1 minute)
**Content**: All indirect, all direct+indirect, minimal evidence
**From**: INTEGRATION_PLAN.md Slide 47
**Key Question**: Do evidence types interact non-linearly?
**Skip if time pressure**: Can mention briefly

### Slide 50: What Remains - Architectures & Italian (1.5 minutes)
**Content**: GPT-2 M/L/XL, LSTM, BERT, n-grams; Italian critical test
**From**: INTEGRATION_PLAN.md Slides 48-49
**Key Point**: Italian should show different evidence hierarchy if theories are right
**Transition**: "There's also the technical challenge of counterfactuals..."

### Slide 51: What Remains - Counterfactuals (45 seconds)
**Content**: English + nulls, Italian + overt, implementation strategy
**From**: INTEGRATION_PLAN.md Slide 50
**Skip if time pressure**: Already covered in Slide 25

### Slide 52: Resource Requirements & Timeline (45 seconds)
**Content**: 500 models, 12K GPU hours, 16 months to chapter
**From**: INTEGRATION_PLAN.md Slide 52
**Figure**: Timeline graphic showing phases
**Transition**: "So what will this contribute?"

---

## PART V: SYNTHESIS (4 minutes)

### Slide 53: Empirical Contributions (1.5 minutes)
**Content**: First systematic test of 40 years of theories; novel findings from pilot; planned complete evidence map
**From**: INTEGRATION_PLAN.md Slide 53
**Key Message**: "This will be the first comprehensive test of acquisition theories using controlled ablation"

### Slide 54: Methodological Contributions (1 minute)
**Content**: New paradigm - controlled rearing, reusable framework, bridges modeling and theory
**From**: INTEGRATION_PLAN.md Slide 54
**Key Message**: "This methodology can be applied to other acquisition phenomena"

### Slide 55: Limitations & Future Directions (1 minute)
**Content**: Models ≠ children (text-only, batch training), future extensions (multimodal, interactive), integration with human studies
**From**: INTEGRATION_PLAN.md Slide 55
**Transition**: "Despite limitations, this addresses fundamental questions..."

### Slide 56: Broader Impact (30 seconds)
**Content**: Understanding human language learning, applications (education, clinical, AI safety), empirical foundation for acquisition science
**From**: INTEGRATION_PLAN.md Slide 56
**Final Message**: "This project provides empirical grounding for theories that have been debated for 40 years"

---

## APPENDIX SLIDES (optional, for Q&A)

### A1: End-State Performance Comparison
**FIGURE**: `analysis/paper_figures/main/endstate_performance.pdf`
**From**: Presentation.tex Appendix

### A2: Developmental Trajectories by Processing Form
**FIGURE**: `analysis/paper_figures/wide/form_trajectories_log.pdf`
**From**: Presentation.tex Appendix

### A3: Learning Trajectories - Expletives
**FIGURE**: `analysis/paper_figures/wide/item_group_trajectories_expletives.png`
**From**: Presentation.tex Appendix

### A4: Learning Trajectories - Control
**FIGURE**: `analysis/paper_figures/wide/item_group_trajectories_control_contexts.png`
**From**: Presentation.tex Appendix

### A5: Learning Trajectories - Long-Distance Binding
**FIGURE**: `analysis/paper_figures/wide/item_group_trajectories_long_distance_binding.pdf`
**From**: Presentation.tex Appendix

### A6: Learning Trajectories - Conjunction
**FIGURE**: `analysis/paper_figures/wide/item_group_trajectories_conjunction.pdf`
**From**: Presentation.tex Appendix

### A7: Model Preferences Table
**FIGURE**: Table from Presentation.tex Appendix

### A8: Age of Acquisition Table
**FIGURE**: `analysis/tables/latex_tables/simple_aoa_table`
**From**: Presentation.tex Appendix

---

## FIGURE SUMMARY TABLE

| Slide | Figure File | Source Location | Purpose |
|-------|------------|-----------------|---------|
| 34 | model_baseline.pdf | main/ | Baseline learning curve |
| 35 | comparison_vs_baseline_overt_only_remove_expletives.pdf | wide/ | Exp 1 results |
| 36 | comparison_vs_baseline_overt_only_impoverish_determiners.pdf | wide/ | Exp 2 results |
| 37 | comparison_vs_baseline_overt_only_remove_articles.pdf | wide/ | Exp 3 results |
| 38 | comparison_vs_baseline_overt_only_lemmatize_verbs.pdf | wide/ | Exp 4 results |
| 39 | comparison_vs_baseline_overt_only_remove_subject_pronominals.pdf | wide/ | Exp 5 results |
| 40 | all_models_comparison_log.pdf | wide/ | Cross-model comparison |
| 42 | forest_form_baseline.pdf | supplementary/ | Processing effects |
| 43 | forms_vs_default_checkmarks (table) | tables/latex_tables/ | Processing table |

**All figures are in**: `../analysis/paper_figures/` (relative to presentations directory)

---

## TIMING CHECKPOINTS

**10 minutes**: Should be at Slide 10 (Hyams' Reversal)
**20 minutes**: Should be at Slide 27 (Timeline - end of Part II)
**30 minutes**: Should be at Slide 35 (Exp 1 - Remove Expletives)
**40 minutes**: Should be at Slide 43 (Processing Effects Table)
**50 minutes**: Should be at Slide 54 (Methodological Contributions)
**60 minutes**: Q&A

---

## KEY SLIDES TO EMPHASIZE (spend extra time)

1. **Slide 18** (The Fundamental Problem) - This is the impasse that motivates everything
2. **Slide 22** (Full Experimental Design) - Shows scope of project
3. **Slide 34** (Baseline Results) - Proof models can learn
4. **Slide 36** (Impoverish Determiners) - Biggest finding
5. **Slide 39** (Remove Pronouns) - Most critical evidence
6. **Slide 40** (Cross-Model Comparison) - Visual evidence hierarchy
7. **Slide 42** (Processing Effects) - Counterintuitive finding
8. **Slide 53** (Empirical Contributions) - What this achieves

---

## SLIDES TO CUT IF RUNNING LONG

Priority cuts to stay on time:
1. Slide 8 (Cluster Properties Examples) - technical detail
2. Slide 14 (Morphological Turn) - can mention briefly
3. Slide 15 (Alternative Perspectives) - can mention briefly
4. Slide 25 (Counterfactual Implementation) - technical detail
5. Slide 31 (Processing Manipulations) - mention in Slide 42
6. Slide 43 (Processing Table) - show in Slide 42
7. Slide 49 (Combined Ablations) - mention briefly in Slide 48
8. Slide 51 (Counterfactuals) - already in Slide 25

**Fast Track** (45 minutes if needed): Cut Slides 8, 14, 15, 25, 31, 43, 49, 51 = 48 slides (~50 min)

---

## TRANSITION SCRIPT HIGHLIGHTS

### Part I → Part II (Slide 19 → 20)
"We can't do these experiments on children. But Small Language Models give us exactly the experimental control we need."

### Part II → Part III (Slide 27 → 28)
"I've already completed Phase 1 as pilot work. Let me show you what I found."

### Part III → Part IV (Slide 46 → 47)
"This pilot proves the approach works and is already revealing. Here's what I'll do next to complete the picture."

### Part IV → Part V (Slide 52 → 53)
"With this full design, here's what we'll contribute to the field."

---

## BACKUP SLIDES FOR ANTICIPATED QUESTIONS

**Q: How do you know models are like children?**
A: Contravariance principle slide (keep from intro), show similar developmental trajectories

**Q: What about other architectures?**
A: Slide 50 (Architectures) + mention this is Phase 2

**Q: How will Italian differ?**
A: Slide 50 (Italian predictions) - should show different evidence hierarchy

**Q: What's the timeline?**
A: Slide 52 (Resources) + Slide 27 (Timeline)

**Q: Can you show more detailed results on X?**
A: Appendix slides A1-A8

---

## NARRATIVE ARC

**Act 1 (Part I)**: The 40-year problem - many theories, no way to test
**Act 2 (Part II)**: The solution - controlled rearing with SLMs  
**Act 3 (Part III)**: The proof - pilot already reveals surprising patterns
**Act 4 (Part IV)**: The plan - systematic completion across languages/architectures
**Act 5 (Part V)**: The payoff - first comprehensive empirical test

---

## PRESENTATION TIPS

1. **Slide 18** is the pivot point - land this hard
2. **Slide 40** is your visual money shot - let it breathe
3. Reference the **figure in Slide 34** when showing later experiments
4. Use **pointer/animation** on forest plot (Slide 42) to show negation effects
5. **Pause** after Slide 36 (determiners) - this is the biggest finding
6. Keep **energy high** through Part III (lots of results) - vary your pacing
7. Have **Appendix ready** for technical questions during Q&A

---

## FIGURE CREATION NEEDS (if not already in files)

May need to create:
1. **Slide 22**: Experimental design flowchart
2. **Slide 27**: Timeline Gantt chart  
3. **Slide 33**: Simple experimental conditions table
4. **Slide 45**: Evidence hierarchy visual (ranking graphic)
5. **Slide 47**: Completed vs. Remaining checklist graphic

All other figures already exist in Presentation.tex paths.
