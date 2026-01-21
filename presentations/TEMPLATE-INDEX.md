# UCSD Brief Template Feature Index

This index provides quick reference to all template features in `ucsd-demo.tex` for AI models to locate and understand presentation components.

## Document Structure

### Preamble & Metadata (Lines 1-32)
**Commands**: `\documentclass`, `\pretitle`, `\title`, `\subtitle`, `\author`, `\date`, `\department`, `\affiliation`, `\footertext`, `\motto`
**When**: Beginning of every document
**Description**: Document setup with title page information and metadata

### Outline Slide (Lines 37-44)
**Command**: `\outline{}`
**When**: After title page to show presentation structure
**Description**: Bulleted list of chapter/section topics

## Basic Slide Types

### Single Column (Lines 47-68)
**Section**: Standard text
**When**: Default slide layout for text and simple content
**Description**: Full-width text with standard paragraphs

### Double Column (Lines 70-89)
**Commands**: `\twocolumn`, `\onecolumn`
**When**: Parallel lists, side-by-side content
**Description**: Text flows from left to right column automatically

### Tables (Lines 91-110)
**Environment**: `tabular` with `\cellcolor`
**When**: Organizing structured information in grid format
**Description**: Custom table layouts with colored headers

### Quad Charts (Lines 112-169)
**Command**: `\quadchart{}{}{}{}`
**When**: Four-quadrant overview (budget, progress, metrics, impact)
**Description**: 2x2 grid for comprehensive project summaries

### Centered Content (Lines 171-175)
**Environment**: `\Center`
**When**: Emphasizing key statements or figures
**Description**: Horizontal and vertical centering

## Code Environments

### Python (Lines 180-198)
**Environment**: `python`
**When**: Python code examples
**Description**: Syntax-highlighted Python with line numbers
**Highlighting**: Use `` `\HL` `` for line highlighting

### MATLAB (Lines 200-221)
**Environment**: `matlab`
**When**: MATLAB/Octave code examples
**Description**: MATLAB-specific syntax highlighting

### R Language (Lines 223-238, commented)
**Environment**: `rlang`
**When**: R statistical code examples
**Description**: R syntax highlighting (commented out in demo)

### Pseudocode (Lines 241-256)
**Environment**: `pseudocode`
**When**: Language-agnostic algorithms
**Description**: Mathematical notation with algorithm structure

## Research Methods

### Study Design (Lines 258-282)
**Sections**: `\section`, `\subsection`
**When**: Describing overall methodology
**Description**: Hierarchical organization of research approach

### Data Management (Lines 284-288)
**Section**: Standard text
**When**: Explaining data handling procedures
**Description**: Data archival and sharing practices

## Advanced Features

### Callout Boxes (Lines 293-304)
**Commands**: `\alertbox{}`, `\infobox{}`, `\highlightbox{}`
**When**: Highlighting important information
**Description**:
- `\alertbox`: Warnings/notes with "Note:" prefix
- `\infobox`: Supplementary information
- `\highlightbox`: Key takeaways

### Theorem Environments (Lines 306-321)
**Environments**: `theorem`, `definition`, `example`
**When**: Formal mathematical statements
**Description**: Numbered theorem-style environments with optional titles

### QR Codes (Lines 323-331)
**Command**: `\qrlink[size]{url}`
**When**: Sharing links to papers, data, repositories
**Description**: Generates QR code with optional size parameter

### Blank Pages (Lines 333-373)
**Command**: `\blankpage`, `\pos{x}{y}{content}`
**When**: Custom graphics or full-page layouts
**Description**: Empty page with TikZ graphics and absolute positioning

## AI-Friendly Features (Chapter 5)

### Structured Content Blocks (Lines 377-399)
**Commands**: `\slidegoal{}`, `\context{}`, `\takeaway{}`, `\source{}`
**When**: Semantic markup of slide purpose
**Description**:
- `\slidegoal`: Objective of slide
- `\context`: Background information
- `\takeaway`: Key message
- `\source`: Citation/attribution

### Key Point Slide (Lines 401-402)
**Command**: `\keypoint{title}{text}`
**When**: Single important statement
**Description**: Large centered key message

### Timeline (Lines 404-415)
**Command**: `\timeline{title}{items}` with `\timelineitem{date}{description}`
**When**: Chronological progression
**Description**: Numbered timeline with dates and events

### Split Slide (Lines 417-441)
**Command**: `\splitslide{title}{left}{right}`
**When**: Side-by-side comparisons
**Description**: 50/50 two-column layout

### Comparison (Lines 443-459)
**Command**: `\comparison{title}{nameA}{contentA}{nameB}{contentB}`
**When**: Contrasting two approaches/methods
**Description**: Side-by-side comparison with subsection headers

### Problem-Solution (Lines 461-490)
**Command**: `\problemsolution{title}{problem}{solution}`
**When**: Presenting challenge and proposed approach
**Description**: Two-section layout with problem statement and solution

### Before-After (Lines 492-518)
**Command**: `\beforeafter{title}{before}{after}`
**When**: Showing change over time
**Description**: Side-by-side comparison of states

### Pros-Cons (Lines 520-534)
**Command**: `\proscons{title}{pros}{cons}`
**When**: Evaluating advantages and disadvantages
**Description**: Two-column itemized lists

### Icon Items (Lines 536-548)
**Command**: `\iconitem{icon}{text}`
**When**: Step-by-step processes with visual markers
**Description**: Icon/number + text layout

### Grid Layout (Lines 550-568)
**Command**: `\gridlayout{title}{topleft}{topright}{bottomleft}{bottomright}`
**When**: Four related topics/areas
**Description**: 2x2 grid layout

### Data Slide (Lines 570-587)
**Command**: `\dataslide{title}{content}`
**When**: Presenting quantitative data/visualizations
**Description**: Marks slide as containing data with alert box

### Simple Table (Lines 575-581)
**Command**: `\simpletable{caption}{columns}{rows}`
**When**: Quick table creation
**Description**: Centered table with caption

### Bar Graph (Lines 589-594)
**Command**: `\bargraph{caption}{labels}{coordinates}`
**When**: Simple bar chart visualization
**Description**: TikZ-based bar chart with coordinates

## Research Presentation Patterns (Chapter 6)

### Multi-Statement Slide (Lines 601-610)
**Command**: `\statements{title}{statement1}{statement2}{statement3}`
**When**: Research motivation with 3 key points
**Description**: Large centered statements with spacing

### Research Questions (Lines 612-623)
**Command**: `\researchquestions{title}{questions}`
**When**: Listing study research questions
**Description**: Alert box header + itemized questions

### Method Slide (Lines 625-645)
**Command**: `\methodslide{title}{approach}{details}`
**When**: Describing experimental methodology
**Description**: Info box with approach + detailed content

### Design Matrix (Lines 647-666)
**Command**: `\designmatrix{title}{tabular content}`
**When**: Experimental conditions/design overview
**Description**: Centered table showing study design

### Results Slide (Lines 668-693)
**Command**: `\resultslide{title}{figure}{caption}{commentary}` OR manual minipage layout
**When**: Presenting results with figure + interpretation
**Description**: Figure left (52%), bullet points right (45%)

### Cross-Model Comparison (Lines 695-716)
**Section**: Standard with centered table
**When**: Comparing multiple conditions/models
**Description**: Table with statistical annotations

### Theoretical Comparison (Lines 718-734)
**Command**: `\comparison{}`
**When**: Contrasting theoretical accounts
**Description**: Reuse of comparison template for theories

### Problem-Solution (Lines 736-765)
**Command**: `\problemsolution{}`
**When**: Theoretical challenge + research approach
**Description**: Reuse for theoretical problems

### Implications (Lines 767-788)
**Section**: `\takeaway{}` + enumerated list + `\source{}`
**When**: Discussing theoretical/practical implications
**Description**: Key takeaway + detailed implications + citation

## Closing Page (Line 790)
**Command**: `\closing`
**When**: End of presentation
**Description**: Final slide with contact information

---

## Quick Reference by Use Case

### Starting a Presentation
- Lines 1-32: Preamble and metadata
- Lines 37-44: Outline slide
- Lines 47-68: First content slide

### Showing Code
- Lines 180-198: Python
- Lines 200-221: MATLAB
- Lines 241-256: Pseudocode

### Presenting Data
- Lines 570-587: Data slide with table
- Lines 589-594: Bar graph
- Lines 668-693: Results with figure

### Theoretical Content
- Lines 306-321: Theorems/definitions
- Lines 718-734: Theoretical comparison
- Lines 736-765: Problem-solution

### Emphasizing Information
- Lines 293-304: Callout boxes
- Lines 401-402: Key point
- Lines 767-788: Takeaway

### Organizing Content
- Lines 70-89: Two columns
- Lines 112-169: Quad chart
- Lines 550-568: Grid layout

### Research Presentations
- Lines 601-610: Motivation
- Lines 612-623: Research questions
- Lines 625-645: Methods
- Lines 647-666: Design matrix
- Lines 668-693: Results
- Lines 767-788: Implications
