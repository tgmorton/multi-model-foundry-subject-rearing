# Null Subject Analysis Script Structure - UPDATED

## Current Script Structure (CLEAN & COMPLETE)

```
analysis/scripts/null_subject_analysis.R (~1700 lines)
├── 1. SETUP & DATA LOADING (lines 1-50) ✅ COMPLETE
│   ├── Libraries (tidyverse, lme4, ggplot2, kableExtra, etc.)
│   ├── Directory creation (organized by model folders + combined)
│   ├── Data loading (conditional check to avoid reloading)
│   ├── Model labels mapping
│   └── Factor conversions
│
├── 2. MAIN OVERVIEW VISUALIZATION (lines 51-110) ✅ COMPLETE
│   ├── Overall preference summary calculation
│   ├── Main preference plot (all models faceted)
│   └── Save to combined/ folder
│
├── 3. ACQUISITION POINT CALCULATION (lines 111-276) ✅ COMPLETE
│   ├── find_crossover_point() function (smart algorithm - early preference)
│   ├── Data summaries for plotting:
│   │   ├── pref_summary (overall)
│   │   ├── itemgroup_summary
│   │   ├── form_summary
│   │   └── form_itemgroup_summary
│   ├── Crossover calculations:
│   │   ├── overall_crossover
│   │   ├── itemgroup_crossover
│   │   ├── form_crossover
│   │   └── form_itemgroup_crossover
│   └── Save acquisition CSVs to tables/
│
├── 4. MODEL COMPARISON FIGURES (lines 277-651) ✅ COMPLETE
│   ├── General model comparison (faceted null|overt)
│   ├── General model comparison (log scale)
│   ├── Combined model comparison (null+overt lines together)
│   ├── Combined model comparison (log scale)
│   ├── Null-only model comparison
│   ├── Null-only model comparison (log scale)
│   └── All saved to combined/ folder
│
├── 5. MODEL COMPARISON BY FORM (lines 652-743) ✅ COMPLETE
│   ├── Models stacked vertically, null|overt horizontally
│   ├── Form-specific acquisition lines
│   ├── Regular and log scale versions
│   └── Saved to combined/ folder
│
├── 6. ACQUISITION TIMING TABLE (lines 744-780) ✅ COMPLETE
│   ├── Simple ranked table of 50/50 acquisition checkpoints
│   ├── CSV and LaTeX outputs
│   └── Console printing
│
├── 7. INDIVIDUAL MODEL FIGURES (lines 781-1180) ✅ COMPLETE
│   ├── Item group figures (per model, with acquisition lines)
│   ├── Form figures (per model, with acquisition lines)
│   ├── Log-transformed versions of both
│   └── All saved to individual model folders
│
├── 8. DETAILED INDIVIDUAL FIGURES (lines 1181-1485) ✅ COMPLETE
│   ├── Simple acquisition (null vs overt, collapsed)
│   ├── Item groups vertical (stacked, null|overt side-by-side)
│   ├── Forms vertical (stacked, null|overt side-by-side)
│   └── All saved to individual model folders
│
├── 9. BASELINE COMPARISONS (lines 1486-1635) ✅ COMPLETE
│   ├── Direct baseline vs manipulation comparisons
│   ├── One per non-baseline model folder
│   ├── Regular and log scale versions
│   └── Shows null+overt for both models
│
├── 10. DESCRIPTIVE STATISTICS TABLES (lines 1636-1780) ✅ COMPLETE
│   ├── calc_preference_stats() helper function
│   ├── Table 1: Overall model preferences
│   ├── Table 2: Model × item group preferences
│   ├── Table 3: Model × individual items
│   ├── Table 4: Model × forms
│   ├── Table 5: Item group × forms
│   └── All saved as CSV files
│
└── 11. COMPLETION & SUMMARY (lines 1781-1790) ✅ COMPLETE
    ├── Status messages
    └── File listing summary
```

## Figure Output Organization

```
analysis/figures/
├── combined/                          # Cross-model comparisons
│   ├── null_overt_preference_by_model.*       # Main overview
│   ├── models_comparison_general.*            # Faceted null|overt
│   ├── models_comparison_general_log.*        # Log scale version
│   ├── models_comparison_combined.*           # Combined null+overt lines
│   ├── models_comparison_combined_log.*       # Log scale version  
│   ├── models_comparison_null_only.*          # Null preference only
│   ├── models_comparison_null_only_log.*      # Log scale version
│   ├── models_comparison_by_form.*            # Form differences within models
│   └── models_comparison_by_form_log.*        # Log scale version
│
├── baseline/                          # Baseline model figures
│   ├── simple_acquisition.*                   # Simple null vs overt
│   ├── itemgroup_acquisition.*               # Item groups faceted
│   ├── itemgroup_acquisition_log.*           # Log scale version
│   ├── itemgroups_vertical.*                 # Item groups stacked
│   ├── form_acquisition.*                    # Forms faceted
│   ├── form_acquisition_log.*                # Log scale version
│   └── forms_vertical.*                      # Forms stacked
│
├── remove_expletives/                 # Remove Expletives model
│   ├── [same 7 figures as baseline]
│   ├── baseline_comparison.*                  # Direct vs baseline
│   └── baseline_comparison_log.*             # Log scale version
│
├── impoverish_determiners/            # Impoverish Determiners model
│   ├── [same 7 figures as baseline]
│   ├── baseline_comparison.*
│   └── baseline_comparison_log.*
│
├── remove_articles/                   # Remove Articles model
│   ├── [same 7 figures as baseline] 
│   ├── baseline_comparison.*
│   └── baseline_comparison_log.*
│
├── lemmatize_verbs/                   # Lemmatize Verbs model
│   ├── [same 7 figures as baseline]
│   ├── baseline_comparison.*
│   └── baseline_comparison_log.*
│
└── remove_subject_pronominals/        # Remove Subject Pronominals model
    ├── [same 7 figures as baseline]
    ├── baseline_comparison.*
    └── baseline_comparison_log.*
```

## Table Outputs

```
analysis/tables/
├── acquisition_points_overall.csv
├── acquisition_points_by_itemgroup.csv  
├── acquisition_points_by_form.csv
├── acquisition_points_by_form_itemgroup.csv
├── acquisition_timing_summary.csv
├── acquisition_timing_summary.tex
├── table1_model_preferences.csv
├── table1_model_preferences.tex
├── table2_model_itemgroup_preferences.csv
├── table3_model_items_preferences.csv
├── table4_model_forms_preferences.csv
└── table5_itemgroup_forms_preferences.csv
```

## Key Features Implemented

### ✅ COMPLETED FEATURES:
1. **Organized file structure** - Model folders + combined folder
2. **Smart acquisition algorithm** - Early preference detection (60% window)
3. **Comprehensive figure types** - 7 types per model + 8 combined types
4. **Statistical rigor** - 95% CIs based on item/form variation
5. **Multiple scales** - Regular and log-transformed versions
6. **Acquisition timing** - Visual markers and summary tables
7. **Direct comparisons** - Baseline vs each manipulation
8. **Complete tables** - All 5 descriptive statistics tables
9. **Consistent styling** - Color schemes, themes, legends
10. **Flexible analysis** - Multiple granularity levels

### 📊 TOTAL OUTPUT:
- **Combined figures**: 8 files (4 comparison types × 2 scales)
- **Individual model figures**: 42 files (6 models × 7 types)
- **Baseline comparisons**: 10 files (5 models × 2 scales)  
- **Tables**: 12 files (acquisition + descriptive statistics)
- **TOTAL**: 72 output files