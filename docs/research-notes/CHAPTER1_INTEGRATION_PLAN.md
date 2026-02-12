# Chapter 1 Integration Plan: From Pilot to Full Proposal

## Strategic Repositioning

### Current State
- **Presentation.pdf**: Contains completed experiments (5 ablations, baseline)
- **Oct22Meeting.tex**: Contains theoretical motivation
- **Goal**: Reframe completed work as pilot data within a larger project proposal

### New Narrative Structure
1. **Part I: Theoretical Motivation** (from Oct22Meeting.tex - Acts 1-3)
2. **Part II: Proposed Research** (NEW - comprehensive project description)
3. **Part III: Pilot Evidence** (from Presentation.pdf - Chapter 1 results)
4. **Part IV: Remaining Work** (from research plan document)

---

## Detailed Slide-by-Slide Plan

### PART I: THEORETICAL MOTIVATION (Keep from Oct22Meeting.tex)
**Slides 1-22**: Maintain current structure (Acts 1-3)
- Opening statement
- The phenomenon (null subject variation)
- Plato's problem
- 40 years of competing theories
- The impasse: can't test on children

---

### PART II: PROPOSED RESEARCH (NEW CONTENT)

#### Slide 23: "Chapter 1: What Evidence Matters for Learning Subject Drop?"
**Content**:
```
Research Questions:
• What information in the linguistic environment contributes to learning 
  constraints around subject drop?
• How much does direct vs. indirect evidence matter?
• What is the most important indirect evidence?
• How do statistical learners solve learning problems without negative evidence?
```

#### Slide 24: "The Ablation Logic"
**Content**:
```
Core Approach: Controlled Rearing Experiments

If we remove evidence type X:
  • Learning fails → X is NECESSARY
  • Learning succeeds → X is SUFFICIENT (but not necessary)
  • Learning delays → X is HELPFUL

Test each theoretical prediction systematically
```

#### Slide 25: "Full Experimental Design Overview"
**Content**:
```
Three Study Types:

1. SINGLE ABLATIONS (systematic isolation)
   • Determiner richness (a/the → DET)
   • Determiner presence (remove a/the entirely)
   • Verbal agreement morphology (lemmatize)
   • Presence of pronouns (remove I/you/he/she...)
   • Expletives (remove it/there)
   • Tense/aspect marking

2. CONTINUOUS SWEEPS
   • Vary subject pronoun : subject drop ratio systematically
   • Map learning curve as function of evidence strength

3. COMBINED ABLATIONS
   • All indirect evidence (determiners + morphology)
   • All direct + indirect (Italian-like)
```

#### Slide 26: "Cross-Linguistic & Cross-Architecture Design"
**Content**:
```
Languages:
• English (non-null-subject)
• Italian (consistent null-subject)
• Counterfactual manipulations:
  - English with inserted null subjects
  - Italian with inserted overt subjects

Architectures:
• Transformers: GPT-2 Small, Medium, Large, XL
• Recurrent: LSTM
• Masked: BERT variants
• Baselines: n-gram models

Multiple runs per condition (3-10) for robustness
```

#### Slide 27: "Evaluation Framework"
**Content**:
```
Behavioral Measures:
• Grammaticality judgments (surprisal on minimal pairs)
• Production preferences (null vs. overt)
• Developmental trajectories (checkpointed training)

Representational Measures:
• Probing classifiers
• Activation similarity
• Cross-linguistic transfer

Statistical Measures:
• Age of Acquisition (AoA) - when cross 50%
• Effect sizes (Cohen's d)
• Learning curves (log-scale splines)
```

#### Slide 28: "Counterfactual Implementation: The Technical Challenge"
**Content**:
```
How to create "English with null subjects"?

Option 1: BERT-based insertion
  • Use masked LM to predict high-probability subjects
  • Preserve discourse coherence

Option 2: Dummy pronoun approach
  • Avoids coreference problem
  • But alters grammatical system

Option 3: Seq2seq model
  • Train on manual examples
  • Need ~1000 paragraph-length samples
  • Can use eval stimuli + manual annotation

Option 4: LLM assistance
  • Use DeepSeek/Claude for annotation
  • Human verification loop
```

#### Slide 29: "Dataset Specifications"
**Content**:
```
Training Corpora (BabyLM):
• ~90M words per language
• Child-directed speech weighted
• Held-out test sets (10M words)
• Ablation replacement sets (10M words)

Evaluation Sets:
• Minimal pairs (null vs. overt)
• Processing manipulations:
  - Negation (context/target/both)
  - Complexity (long NPs, embeddings)
• Test families (expletives, control, etc.)
• ~300 items per language
```

#### Slide 30: "Timeline & Pre-Registration"
**Content**:
```
Phase 1: Pilot (COMPLETED - see next section)
  ✓ 5 single ablations + baseline
  ✓ English only
  ✓ GPT-2 Small
  ✓ BabyLM corpus

Phase 2: Single Ablations (In Progress)
  • Complete remaining ablations (tense/aspect)
  • Add Italian
  • Add architectural variants

Phase 3: Continuous Sweeps (Planned)
  • Subject pronoun ratio manipulation
  • Determiner richness gradients

Phase 4: Combined Ablations (Planned)
  • Interactions between evidence types

Pre-registration: OSF (before Phase 2 completion)
```

---

### PART III: PILOT EVIDENCE (From Presentation.pdf)

#### Slide 31: "Pilot Study: 5 Ablation Experiments"
**Content**:
```
What We've Done:
✓ Baseline: Full BabyLM corpus
✓ Remove Expletives: No it/there constructions
✓ Impoverish Determiners: a/the → DET
✓ Remove Articles: No a/the entirely
✓ Lemmatize Verbs: Remove -s/-ed/-ing
✓ Remove Subject Pronouns: No I/you/he/she/it/we/they

Language: English
Architecture: GPT-2 Small (124M parameters)
Training: 5 epochs, ~450M tokens
Evaluation: 300 minimal pairs across 6 test families
```

#### Slide 32: "Materials: BabyLM Dataset"
**Content**: From Presentation slide 19
```
Training Corpus
• 90M word corpus designed for human-sized models
• Linguistically diverse with child-directed speech
• Models linguistic input of 10-14 year old child
• 10M word held-out test set
• 10M word ablation replacement set

Dataset Composition
• CHILDES (child-directed speech): 29M words
• Project Gutenberg (children's stories): 26M words
• OpenSubtitles (movie subtitles): 20M words
• Simple English Wikipedia: 15M words
• BNC dialogue + Switchboard: 9M words
```

#### Slide 33: "Evaluation: Minimal Pairs Design"
**Content**: From Presentation slide 20
```
Core Contrasts (English non-pro-drop)
• Person/Number: Anna finished. She/*∅ thinks...
• Control: Maria convinced her brother ∅/*him to leave
• Expletives: *∅/It seems that students passed
• Topic shift: Anna called Mark and *∅/he refused

Minimal Pairs Design
• Sentences differ only in subject realization
• Lexical and contextual content held constant
• 6 test families
• Tests both grammatical competence and processing effects
```

#### Slide 34: "Statistical Approach"
**Content**: From Presentation slides 16-18
```
Outcome Measure:
• Binary: overt preference when P(overt) > P(null)
• Calculated via surprisal on minimal pairs

Modeling:
• Logistic GLMMs with natural splines
• Log-scale checkpoints (reflecting NN learning dynamics)
• Random effects by item

Key Metrics:
• Age of Acquisition (AoA₅₀): Last crossing of 50%
• AoA₁/₂: Halfway to asymptote
• Bootstrap 95% CIs (n=500)
• Effect sizes: ΔAoA between conditions
```

#### Slide 35: "Pilot Result 1: Universal Null-Subject Stage"
**Content**: From Presentation slide 33
```
Surprising Finding: ALL models start with null preference

0 10 100 1K 10K
Training Step (Log)
[Figure showing all models starting >60% null preference]

Theoretical Implications:
• Consistent with child acquisition (null-first accounts)
• Contradicts Bloom's prediction of overt-first
• Could reflect:
  - Model architecture bias, OR
  - Environmental evidence (discourse drops in training data)

Question: Is this learning bias or input-driven?
→ Needs Italian comparison + counterfactual English
```

#### Slide 36: "Pilot Result 2: Baseline Developmental Trajectory"
**Content**: From Presentation slide 22
```
Baseline Model - Training Curves
[Figure from slide 22]

Key Findings:
• Initial null preference: 63.4% (95% CI [62.7, 64.1])
• Age of Acquisition: checkpoint 727 (95% CI [664, 791])
• Final overt preference: 69.6% (95% CI [66.5%, 72.5%])

Pattern mirrors human acquisition:
• Start with null subjects
• Gradual shift to overt
• Stabilize at adult-like preference
```

#### Slide 37: "Pilot Result 3: Expletives Are Helpful But Not Critical"
**Content**: From Presentation slide 23
```
Exp 1: Remove Expletives
[Figure showing delayed but successful learning]

Age of Acquisition: 767 (95% CI [709, 821])
Delay: +39 epochs vs. baseline (p<.001)

Interpretation:
• Yang's prediction: Expletives are crucial
• Result: Learning succeeds without expletives
• Expletives accelerate learning but are not necessary

Final performance: No significant difference from baseline
```

#### Slide 38: "Pilot Result 4: Determiners Provide Crucial Shortcuts"
**Content**: From Presentation slide 24
```
Exp 2: Impoverish Determiners (a/the → DET)
[Figure showing dramatic delay]

Age of Acquisition: 3400 (95% CI [3307, 3499])
Delay: +2672 epochs vs. baseline (p<.001)

Interpretation:
• Duguine's prediction: Rich determiners matter
• Result: MASSIVE delay when determiners impoverished
• Eventually achieves STRONGEST overt preference (grokking?)

Removing determiners forces slower but potentially deeper learning
```

#### Slide 39: "Pilot Result 5: Articles Matter More Than Expletives"
**Content**: From Presentation slide 25
```
Exp 3: Remove Articles Entirely
[Figure]

Age of Acquisition: 807 (95% CI [758, 861])
Delay: +80 epochs vs. baseline (p<.001)

Comparison:
• Remove articles: +80 epochs delay
• Remove expletives: +39 epochs delay
• Articles have ~2x the impact of expletives

Stronger initial null preference (71.7%)
Lower final overt preference (68.2%) than baseline
```

#### Slide 40: "Pilot Result 6: Morphology Paradox"
**Content**: From Presentation slide 26
```
Exp 4: Lemmatize Verbs (remove -s/-ed/-ing)
[Figure]

Age of Acquisition: 705 (95% CI [660, 748])
Speedup: -22 epochs vs. baseline (p=.034)

FASTEST acquisition among all interventions

Interpretation Challenge:
• Hyams' prediction: Non-uniform morphology triggers learning
• Result: Removing morphology ACCELERATES learning
• Paradox: Morphology seems to interfere, not help

Possible explanation: Morphology adds noise to the learning signal
```

#### Slide 41: "Pilot Result 7: Direct Evidence Is Critical"
**Content**: From Presentation slide 27
```
Exp 5: Remove Subject Pronouns
[Figure]

Age of Acquisition: 774 (95% CI [706, >5000])
Wide CI indicates unstable learning

Final overt preference: 54.4% (near chance)
WEAKEST performance of all models

Interpretation:
• Direct evidence (overt pronouns) is NECESSARY
• Without pronouns, model barely learns
• Supports Hyams' direct evidence account
• Indirect evidence alone is insufficient
```

#### Slide 42: "Pilot Result 8: Evidence Hierarchy"
**Content**: From Presentation slide 34 + cross-model comparison
```
Rank Order of Learning Success (by AoA):
1. Lemmatize Verbs: 705 (fastest)
2. Baseline: 727
3. Remove Expletives: 767
4. Remove Pronouns: 774 (unstable)
5. Remove Articles: 808
6. Impoverish Determiners: 3400 (slowest by far)

Evidence Effectiveness:
CRITICAL: Subject pronouns (direct evidence)
HELPFUL: Articles, determiners (shortcuts to generalization)
MINIMAL: Expletives (slight acceleration only)
INTERFERING(?): Verbal morphology (slows learning)
```

#### Slide 43: "Pilot Result 9: Processing Effects"
**Content**: From Presentation slides 29-31
```
Processing Manipulations:
• Negation (context/target/both)
• Complex syntax (long NPs, embeddings)

Human Prediction (Bloom 1990): 
Processing load → MORE null subjects (omission under resource limits)

Model Result: OPPOSITE
Processing load → MORE overt subjects

[Table from slide 31]
Negation: Universally increases overt preference (all models)
Complexity: Largely no effect

Interpretation:
• Models behave opposite to processing account
• Processing theories may not fully explain child null subjects
• Need empirical validation in human production studies
```

#### Slide 44: "Pilot Summary: Theoretical Implications"
**Content**: From Presentation slide 35
```
What the Pilot Challenges:
• Yang's expletive-centric account (learning succeeds without)
• Simple parameter-setting (gradual, evidence-based learning)
• Morphological triggers (removal speeds learning)
• Processing accounts (opposite pattern in models)

What the Pilot Supports:
• Hyams' direct evidence account (pronouns critical)
• Duguine's determiner hypothesis (major impact)
• Gradual statistical learning
• Convergence via Bayesian inference

Open Questions:
• Why universal null-first stage?
• Why does morphology interfere?
• Will patterns replicate in Italian?
• Do architectural differences matter?
```

---

### PART IV: REMAINING WORK (From Research Plan)

#### Slide 45: "What Remains: Single Ablations"
**Content**:
```
Completed (English):
✓ Determiners (richness + presence)
✓ Verbal morphology
✓ Pronouns
✓ Expletives

Still Needed:
□ Tense/aspect marking (separate from agreement)
□ Subject drop presence (insert nulls in English training)
□ Replicate all above in Italian
□ Test counterfactual manipulations:
  - English + inserted null subjects
  - Italian + inserted overt subjects
```

#### Slide 46: "What Remains: Continuous Sweeps"
**Content**:
```
Goal: Map learning as function of evidence strength

Continuous Manipulations:
□ Subject pronoun : null subject ratio
  - English: 100:0 (baseline) → 70:30 → 50:50 → 30:70 → 0:100
  - Creates gradient of direct evidence

□ Determiner richness gradient
  - Full system (a/the/this/that/these/those)
  - Reduced (a/the)
  - Impoverished (DET)
  - None

□ Morphology gradient
  - Full paradigm
  - Reduced (only -s present)
  - Uniform (all forms identical)

Expected: Sigmoidal learning curves with identifiable thresholds
```

#### Slide 47: "What Remains: Combined Ablations"
**Content**:
```
Test Interactions Between Evidence Types:

1. All Indirect Evidence (English + Italian)
   • Remove: pronouns (direct evidence)
   • Keep: determiners, morphology, expletives
   • Question: Can indirect evidence suffice?

2. All Direct + Indirect (Italian only)
   • Keep: pronouns
   • Keep: all indirect evidence
   • Question: Maximal evidence condition

3. Minimal Evidence (English + Italian)
   • Remove: pronouns, determiners, morphology
   • Keep: Only word order, lexical items
   • Question: What's the lower bound?

Predictions:
• Interactions may be non-linear
• Some combinations may show redundancy
• Others may show necessity of joint presence
```

#### Slide 48: "What Remains: Architectural Comparisons"
**Content**:
```
Completed:
✓ GPT-2 Small (124M parameters)

Planned:
□ GPT-2 Medium (355M)
□ GPT-2 Large (774M)
□ GPT-2 XL (1.5B)
□ LSTM (matched capacity)
□ BERT base/large (masked LM)
□ N-gram baselines (3-gram, 5-gram)

Questions:
• Does scale matter? (capacity × data interaction)
• Do architectural biases differ?
  - Autoregressive (GPT-2, LSTM)
  - Masked (BERT)
  - Non-neural (n-grams)
• Which architectures show human-like patterns?
```

#### Slide 49: "What Remains: Italian Experiments"
**Content**:
```
Why Italian Matters:
• Tests if patterns generalize cross-linguistically
• Null-subject language: opposite learning problem
• Should show MAINTENANCE of null preference (not acquisition of overt)

Italian Predictions:
Baseline: High null preference throughout (>70%)

Ablations:
• Remove pronouns: Should have MINIMAL effect
  (pronouns less informative in pro-drop language)
• Impoverish determiners: May matter MORE
  (if Duguine is right about inverse approach)
• Morphology: Should matter MORE
  (rich agreement licenses pro-drop)

Critical Test: Cross-linguistic dissociation of evidence types
```

#### Slide 50: "What Remains: Counterfactual Implementations"
**Content**:
```
Technical Challenge: Create "impossible" languages

English + Null Subjects:
• Insert null subjects at Italian-like rates (~70%)
• Maintain English word order, morphology
• Question: Do models learn "wrong" grammar?

Italian + Overt Subjects:
• Insert overt subjects at English-like rates (100%)
• Maintain Italian word order, morphology
• Question: Do models suppress pro-drop?

Implementation Strategy:
1. Dependency parsing to identify null subject sites
2. BERT-based insertion for appropriate pronouns
3. Human verification (sample 10%)
4. Corpus statistics validation

Expected: ~2-3 months development + validation
```

#### Slide 51: "Statistical Power & Pre-Registration"
**Content**:
```
Sample Size Planning:
• Multiple runs per condition: 3-10
• Pilot power analysis based on completed experiments
• Expected effect sizes (Cohen's d):
  - Large: ΔAoA > 100 epochs (determiners)
  - Medium: ΔAoA 50-100 epochs (articles)
  - Small: ΔAoA < 50 epochs (expletives)

Pre-Registration (OSF):
• Hypotheses for each ablation
• Analysis plan (AoA, learning curves)
• Exclusion criteria
• Multiple comparison corrections

Timeline:
• Pre-register before completing Phase 2
• Ensures confirmatory (not exploratory) testing
```

#### Slide 52: "Resource Requirements"
**Content**:
```
Computational:
• Per model: ~24 GPU-hours (V100)
• Full design: ~500 models
• Total: ~12,000 GPU-hours
• Cost estimate: $15,000-20,000 (cloud compute)

Personnel:
• RA support for data validation (10 hrs/week × 6 months)
• Manual annotation for counterfactuals (40 hrs)

Timeline:
• Phase 2 (remaining single ablations): 3 months
• Phase 3 (continuous sweeps): 4 months
• Phase 4 (combined ablations): 3 months
• Writing + revision: 6 months
• Total: 16 months to dissertation chapter
```

---

### PART V: SYNTHESIS

#### Slide 53: "Contributions: Empirical"
**Content**:
```
First systematic test of 40 years of acquisition theories

Novel Findings (Pilot):
• Direct evidence (pronouns) is NECESSARY
• Determiners provide powerful shortcuts
• Expletives have minimal impact
• Morphology may interfere (not help)
• Universal null-first stage
• Processing effects opposite to predictions

Planned Contributions:
• Complete evidence map (all ablations × both languages)
• Identify interactions between evidence types
• Establish sufficiency vs. necessity distinctions
• Cross-linguistic validation
• Architectural generalizability
```

#### Slide 54: "Contributions: Methodological"
**Content**:
```
New Paradigm: Controlled Rearing for Language Acquisition

Advantages:
• Systematic ablation impossible in children
• Precise control over input
• Multiple runs for robustness
• Transparent representations
• Developmental trajectories

Reusable Framework:
• Can apply to other acquisition phenomena
• Can test other grammatical parameters
• Can investigate learning mechanisms

Impact:
• Bridge computational modeling and acquisition theory
• Generate testable predictions for human studies
• Constrain theoretical debates
```

#### Slide 55: "Limitations & Future Directions"
**Content**:
```
Current Limitations:
• Models ≠ children (important to remember)
• Text-only input (no prosody, visual context)
• Batch training (not interactive)
• No social/pragmatic learning
• Monolingual → multilingual sequencing artificial

Future Extensions:
• Multimodal models (text + visual grounding)
• Interactive learning environments
• Social learning signals
• Broader grammatical phenomena
• Typologically diverse languages
• Child-inspired architectures

Integration with Human Studies:
• Use model predictions to design experiments
• Test specific evidence types in corpus analyses
• Validate findings with human participants
```

#### Slide 56: "Broader Impact: Understanding Human Language Learning"
**Content**:
```
Fundamental Question: How do children learn language from limited input?

This Work Addresses:
• What evidence matters most?
• How do learners use indirect evidence?
• Can positive evidence alone suffice?
• What are the learning mechanisms?

Applications:
• Educational interventions (literacy, L2 learning)
• Clinical assessment (language disorders)
• AI safety (understanding inductive biases)
• Theoretical linguistics (constraining theories)

Vision: Empirical foundation for acquisition science
```

---

## Implementation Notes

### File Organization
```
presentations/
├── CHAPTER1_INTEGRATION_PLAN.md (this file)
├── Oct22Meeting_v2.tex (new version)
├── figures/
│   ├── baseline_curves.pdf
│   ├── exp1_expletives.pdf
│   ├── exp2_determiners.pdf
│   ├── exp3_articles.pdf
│   ├── exp4_morphology.pdf
│   ├── exp5_pronouns.pdf
│   ├── cross_model_comparison.pdf
│   └── processing_effects.pdf
└── tables/
    ├── aoa_summary.tex
    ├── evidence_hierarchy.tex
    └── remaining_work.tex
```

### LaTeX Structure Changes

```latex
% New structure
\part{Theoretical Motivation}
% Current Acts 1-3

\part{Proposed Research}
% Slides 23-30 (NEW)

\part{Pilot Evidence}
% Slides 31-44 (from Presentation.pdf)

\part{Remaining Work}
% Slides 45-52 (from research plan)

\part{Synthesis}
% Slides 53-56 (NEW)
```

### Figure Extraction Needs
From Presentation.pdf, extract:
- Slide 22: Baseline training curves
- Slide 23: Expletives comparison
- Slide 24: Determiners comparison  
- Slide 25: Articles comparison
- Slide 26: Morphology comparison
- Slide 27: Pronouns comparison
- Slide 28: Cross-model comparison
- Slide 31: Processing effects table

### Priority Order for Implementation
1. **Phase 1** (Core slides): 23-30, 31-32, 44 (proposal + pilot intro/summary)
2. **Phase 2** (Results): 33-43 (detailed pilot findings)
3. **Phase 3** (Future): 45-52 (remaining work)
4. **Phase 4** (Synthesis): 53-56 (contributions)

### Presentation Variants
Consider creating:
- **Short version** (30 min): Parts I + II + Part III summary + Part V
- **Medium version** (45 min): Parts I + II + Part III (selected results) + Part IV (overview) + Part V
- **Full version** (60 min): All parts

---

## Key Messaging

### For Advisor
"The pilot demonstrates feasibility and reveals surprising patterns. The full design will systematically test 40 years of theory."

### For Committee
"This is methodologically novel controlled-rearing paradigm with immediate empirical payoff from pilot data."

### For Conference
"We use SLMs as experimental systems to test what evidence children might use for parameter setting."

### For Paper
"First systematic empirical test of competing acquisition theories using controlled ablation experiments."
