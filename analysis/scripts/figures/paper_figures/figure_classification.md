# Figure Classification for kaobook Paper

## Main Text Figures (Regular Width)
**Specs**: 4.21" × 2.6", placed in text column with caption in margin
- Figure 1: Overall preference plot (single model overview)
- Figure 2: Learning curves for baseline model only
- Figure 3: Acquisition timing comparison (t50 across models)
- Figure 4: End-state performance comparison
- Figure 5: Key construction-specific results (subset of most important)

## Wide Figures (Full Width)
**Specs**: 6.48" × 3.24", span text + margin with caption below
- Figure 6: Complete model comparison learning curves (all models)
- Figure 7: Multi-panel hotspot analysis (3×2 grid)
- Figure 8: Theory evaluation matrix (comprehensive comparison)
- Figure 9: Item group acquisition patterns (faceted by groups)
- Figure 10: Chang & Bergen AoA analysis (forest plot + curves)

## Supplementary Figures
**Specs**: Various sizes, go in appendix/supplementary material
- S1-S6: Individual model detailed analyses
- S7-S12: Construction-specific breakdowns
- S13-S18: Hotspot analyses per model
- S19-S24: Diagnostic plots and residual analyses

## Margin Figures
**Specs**: 1.94" × 2.33", embedded in text margins
- Small inset plots showing specific effects
- Method illustrations
- Simple bar charts or line plots for emphasis

## Tables
- Table 1: Model specifications and interventions
- Table 2: Summary statistics (descriptive)
- Table 3: Acquisition timing results (t50, AoA½)
- Table 4: End-state performance
- Table 5: Theory predictions vs. results

## LaTeX Integration Notes
- Regular figures: `\begin{figure}` (margin captions automatic)
- Wide figures: `\begin{figure*}` or `\begin{widefigure}`
- Margin figures: `\begin{marginfigure}`
- Font sizes automatically scale with paper theme
- Colors optimized for both print and digital viewing