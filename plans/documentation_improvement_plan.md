# Documentation Improvement Plan: Preprocessing Module

## Executive Summary

After reviewing all 7 preprocessing documentation files (~2,800 lines total), I've identified key opportunities to make them more human-centric, readable, and user-friendly without sacrificing functionality. The current docs are comprehensive but suffer from excessive technical detail, redundant information, and lack of clear narrative flow.

## Problems Identified

### 1. Information Overload
- **Line counts everywhere**: "500+ lines", "600+ lines" adds noise without value
- **Excessive metadata**: Version numbers, timestamps, and technical details clutter examples
- **Redundant content**: Same concepts explained multiple times across different docs
- **Over-documentation**: Phase changelog and technical minutiae don't help users

### 2. Poor Information Architecture
- **Flat structure**: Docs don't guide users through learning progression
- **Scattered use cases**: Examples spread across multiple files without clear purpose
- **Unclear audience**: Mixing beginner tutorials with advanced technical specs
- **Weak navigation**: Links to other docs without explaining when/why to read them

### 3. User Experience Issues
- **Technical-first language**: Focuses on implementation details before user value
- **Assumed knowledge**: Doesn't explain why linguistic ablations matter
- **Missing quick wins**: Hard to find "just make it work" path
- **Emoji overuse**: ✅ and ⚠️ everywhere creates visual noise

### 4. Content Quality
- **Verbose examples**: Too much boilerplate for simple concepts
- **Weak explanations**: What ablations do, but not why they matter
- **Missing context**: Research use cases and implications unexplained
- **Developer friction**: Template is good but lacks practical guidance

## Proposed Solution: Three-Tier Documentation

### Tier 1: Getting Started (Beginner)
**Audience**: First-time users who want quick results

**Files to create/refactor**:
- `GETTING_STARTED.md` - Replace current README.md
  - What is this? (30 seconds to understand)
  - Quick start (5 minutes to first result)
  - Common tasks (what you'll likely want to do)
  - Where to go next (based on your needs)

### Tier 2: Working Guides (Intermediate)
**Audience**: Users doing real work

**Files to refactor**:
- `GUIDE.md` - Consolidate USER_GUIDE.md and common workflows
  - Understanding ablations (why they matter for research)
  - Using the pipeline (practical patterns)
  - Handling common issues (troubleshooting)

- `CUSTOM_ABLATIONS.md` - Simplified DEVELOPER_GUIDE.md
  - Adding your own ablation (streamlined steps)
  - Real examples with explanations
  - Common patterns and pitfalls

### Tier 3: Advanced Topics (Expert)
**Audience**: Power users needing specialized features

**Files to refactor**:
- `ADVANCED.md` - Merge ADVANCED_USAGE.md + PHASE4_ENHANCEMENTS.md
  - Coreference resolution
  - Performance tuning
  - Error handling strategies
  - Production deployment

### Tier 4: Reference (As-needed)
**Audience**: Looking up specific information

**Files to create**:
- `API_REFERENCE.md` - Extract from scattered docs
  - Configuration options (complete list)
  - Available ablations (with examples)
  - Error codes and meanings

- `TESTING.md` - Keep but simplify
  - Running tests
  - Writing new tests
  - CI/CD integration

### Files to Archive/Delete
- `TEST_STATUS.md` - Delete (put status in TESTING.md)
- `PHASE4_ENHANCEMENTS.md` - Archive (merge content into ADVANCED.md)
- Changelog in README - Delete (use git history)

## Specific Improvements by Document

### README.md → GETTING_STARTED.md

**Remove**:
- All line counts and metadata
- Directory structure diagrams
- Architecture diagrams
- Changelog section
- Benefits comparison tables
- Provenance JSON examples

**Add**:
- Clear 1-paragraph "What is this?"
- Immediate code example with explanation
- Decision tree: "Which ablation do I need?"
- Visual before/after examples

**Restructure**:
```markdown
# Preprocessing: Linguistic Ablations for Text Corpora

Transform text corpora by systematically removing or modifying linguistic features to test how models learn language.

## What This Does

Removes specific language features (articles, pronouns, verb forms) from text to create controlled experiments. For example, remove all articles to test if models can learn grammar without "the", "a", or "an".

## Quick Start

[Minimal example - 5 lines max]

## Choose Your Ablation

[Simple decision tree or flowchart]

## Next Steps

[Based on user goals, not document structure]
```

### USER_GUIDE.md → GUIDE.md

**Remove**:
- Redundant basic usage section (covered in Getting Started)
- Configuration reference (move to API_REFERENCE.md)
- All technical field listings
- Duplicate examples

**Add**:
- Why ablations matter (research context)
- Real research scenarios
- Practical decision-making guidance

**Restructure**:
- Research context first (why this matters)
- Task-oriented sections (by goal, not by ablation type)
- Progressive complexity (simple → complex)

### DEVELOPER_GUIDE.md → CUSTOM_ABLATIONS.md

**Remove**:
- Time estimates (creates pressure)
- Numbered checklists (use narrative flow)
- All emoji
- Basic template (move to appendix)

**Add**:
- Conceptual overview (how the system works)
- Design philosophy (why it's built this way)
- Decision guide (when to create custom vs use existing)

**Restructure**:
- Understand the system
- Design your ablation (thinking process)
- Implement it (with examples)
- Test and deploy

### ADVANCED_USAGE.md + PHASE4_ENHANCEMENTS.md → ADVANCED.md

**Remove**:
- PHASE4 branding (just explain the features)
- Performance comparison tables (generic guidance instead)
- Migration sections (not needed going forward)
- Technical implementation details

**Add**:
- When you need advanced features (decision criteria)
- Trade-off explanations (speed vs accuracy)
- Production deployment patterns

**Restructure by use case**:
- Large-scale processing (performance)
- High-accuracy needs (coreference, validation)
- Production environments (error handling, monitoring)

### TESTING.md

**Remove**:
- NumPy incompatibility section (outdated context)
- Test file listings with counts
- Detailed fixture documentation

**Add**:
- Why testing matters for this domain
- Quick start for common scenarios
- CI/CD integration examples

**Restructure**:
- Running tests (for users)
- Writing tests (for contributors)
- Troubleshooting

### TEST_STATUS.md → DELETE

**Why**: Belongs in TESTING.md as a subsection. Separate doc creates navigation overhead.

**Action**: Move summary table to TESTING.md introduction.

## Writing Style Guide

### Before (Current Style)
```markdown
### AblationConfig Fields

```python
# Required
type: str                           # Ablation type (registered name)
input_path: Path                    # Input corpus directory
output_path: Path                   # Output directory

# Reproducibility
seed: int = 42                      # Random seed
```

**Total: 72 tests passing** ✅
```

### After (Human-Centric Style)
```markdown
### Configuration

Specify what to ablate and where to find your data:

```python
config = AblationConfig(
    type="remove_articles",       # Which ablation to apply
    input_path="data/raw/",       # Your input corpus
    output_path="data/processed/" # Where to save results
)
```

All ablations use the same configuration pattern. [See complete options →](API_REFERENCE.md#configuration)
```

### Key Principles

1. **User value first**: Lead with what users can do, not how it works
2. **Progressive disclosure**: Start simple, reveal complexity as needed
3. **Natural language**: Write like explaining to a colleague
4. **Concrete examples**: Show real use cases, not abstract patterns
5. **Visual hierarchy**: Use structure, not emoji, for emphasis
6. **Active voice**: "Remove articles" not "Articles are removed"
7. **Purposeful links**: Explain why to follow, not just where it goes

## Implementation Plan

### Phase 1: Foundation (Priority 1)
1. Create `GETTING_STARTED.md` (new user experience)
2. Create `API_REFERENCE.md` (consolidate scattered reference material)
3. Delete `TEST_STATUS.md` (merge into TESTING.md)

### Phase 2: Core Guides (Priority 2)
4. Refactor `USER_GUIDE.md` → `GUIDE.md`
5. Refactor `DEVELOPER_GUIDE.md` → `CUSTOM_ABLATIONS.md`
6. Simplify `TESTING.md`

### Phase 3: Advanced (Priority 3)
7. Merge `ADVANCED_USAGE.md` + `PHASE4_ENHANCEMENTS.md` → `ADVANCED.md`
8. Archive old files to `docs/preprocessing/archive/`

### Phase 4: Polish (Priority 4)
9. Update all cross-references
10. Add visual aids (diagrams, flowcharts)
11. Final review for consistency

## Success Criteria

### Quantitative
- Reduce total doc word count by 30-40%
- Reduce time-to-first-success for new users to < 5 minutes
- Reduce number of files from 7 to 5

### Qualitative
- New user can understand what this is in 30 seconds
- User can find relevant information in 2 clicks max
- Each doc has clear audience and purpose
- Zero redundancy between documents
- Professional tone without being dry

## Migration Strategy

### For Users
- Keep old docs accessible in archive/ with redirect notices
- Add "Looking for old docs?" section in GETTING_STARTED.md
- Maintain all existing code examples (in cleaner form)

### For Maintainers
- Update STRUCTURE.md with new organization
- Add contribution guide for docs
- Document style guide for future updates

## Document Structure Summary

**New structure** (5 core docs):
```
docs/preprocessing/
├── GETTING_STARTED.md    # Start here (beginner)
├── GUIDE.md              # Core usage (intermediate)
├── CUSTOM_ABLATIONS.md   # Extending (intermediate)
├── ADVANCED.md           # Optimization & production (expert)
├── API_REFERENCE.md      # Look up specifics (reference)
├── TESTING.md            # Test guide (contributor)
└── archive/              # Old versions (for reference)
    ├── README.md
    ├── USER_GUIDE.md
    ├── DEVELOPER_GUIDE.md
    ├── ADVANCED_USAGE.md
    └── PHASE4_ENHANCEMENTS.md
```

## Next Steps

1. **Review this plan** with stakeholders
2. **Approve scope** (all phases or subset?)
3. **Begin Phase 1** (foundation documents)
4. **Iterate** based on feedback

This plan prioritizes user experience while maintaining technical accuracy and completeness. Every change serves the goal of making the docs more accessible without losing functionality.
