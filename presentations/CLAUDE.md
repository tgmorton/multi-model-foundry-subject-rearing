# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a LaTeX presentation workspace. You will act as an agent assistant for creating and editing LaTeX presentations (`.tex` files).

## Key Resources

- **TEMPLATE-INDEX.md**: Contains a catalog of slide templates with descriptions and line numbers for selective retrieval from the example presentation
- **ucsd-demo.tex**: Comprehensive example presentation containing various layouts and content-rich slides that serve as templates

## Workflow for Creating Presentations

### Multi-Faceted Directives (e.g., "Turn this document into a presentation")

When given complex requests to create presentations:

1. **Content Translation**: First, adapt the provided content into presentation-appropriate format
   - Target the specified audience
   - If no audience is specified, assume a fairly general audience

2. **Slide Breakdown**: Map the translated content onto slides using either:
   - Existing slide templates from the example presentation (referenced via TEMPLATE-INDEX.md)
   - Custom slide formats when appropriate (use creativity and insight into visual communication)

### Single Slide or Edit Requests

When asked to make edits or create individual slides:

- Follow explicit formatting directions if provided
- Use specified templates if directed
- Otherwise, exercise judgment to select appropriate formatting and templates

## Working with Templates

- Reference TEMPLATE-INDEX.md to locate specific slide layouts by line number
- Extract relevant template slides from ucsd-demo.tex as needed
- Adapt templates to fit the specific content and communication goals

## Important Workflow Notes

- **Assume concurrent editing** - The user is actively making changes to documents at the same time
  - If you encounter "file has been modified" errors, simply re-read the file and retry the edit
  - This is normal collaborative workflow, not an error condition

## LaTeX Compilation

- **LaTeX Installation Path**: `/usr/local/texlive/2023/bin/universal-darwin`
- **Compilation Workflow**:
  1. Compile with: `pdflatex -interaction=nonstopmode <filename>.tex 2>&1 | grep -A 5 "^!"`
  2. Extract and display only errors (lines starting with `!`)
  3. Fix errors recursively until compilation succeeds
  4. Common fixes:
     - Missing packages: Install via `tlmgr install <package>`
     - Syntax errors: Fix LaTeX code in the .tex file
     - Missing dependencies: Check error messages for required files
