# Preprocessing Documentation Changes

## 2026-04 — Three-Tier English Expletive Detector

The `remove_expletive_sentences_en` ablation has been upgraded from a
single-line `dep_ == 'expl'` check into a **stateful three-tier detector**
implemented as a callable class
(`EnglishExpletiveSentenceRemover`,
`preprocessing/ablations/remove_expletive_sentences.py:57`).

### Detection tiers

1. **Tier 1** — spaCy `dep_ == 'expl'` (existential-*there*, expletive-*it*).
2. **Tier 2** — heuristic weather / raising-*it*:
   - `tier2_weather` — "it" as `nsubj` of a verb in `WEATHER_VERBS`.
   - `tier2_raising` — "it" as `nsubj` of a raising verb with a clausal
     complement.
   - `tier2_copular` — copula + adjective in `RAISING_ADJECTIVES` with a
     clausal complement.
3. **Tier 3** — optional coreference confirmation. When `coref_model` is
   set, referential *it* is kept (`tier3_coref_kept`), otherwise the line
   is removed (`tier3_coref_confirmed`).

Document boundary markers (`= = = ... = = =`) are detected and pass
through untouched, also resetting the coref context buffer.

### Per-tier provenance plumbing

`AblationPipeline._process_file` (`preprocessing/base.py:257`) now looks
for two optional hooks on the registered ablation:

- `reset_file_state()` — called at the start of each file.
- `get_file_tier_counts()` — returns a `Dict[str, int]` of per-tier
  removal counts for the file.

If the ablation exposes `_removed_line_indices`, those indices are also
captured into `FileStatistics`.

`FileStatistics` (`preprocessing/config.py:221`) and `ProvenanceMetadata`
(`preprocessing/config.py:120`) grew new fields:

- `tier_counts`, `removed_line_indices`
- `replacement_pool_size`, `replacement_lines_drawn`,
  `replacement_pool_remainder`
- `aggregate_tier_counts`, `total_pool_lines_{available,drawn,remaining}`

### Replacement-pool stats

`_rebuild_to_target_size` now returns pool statistics alongside the
rebuilt text (`preprocessing/base.py:433`) so the manifest records how
many lines were drawn from the pool and how many remained.

### Experiment-level seed fallback

`model_foundry/cli.py:135` auto-injects the experiment-level `random_seed`
into preprocessing step `parameters` if the step does not set `seed`
itself, so experiments remain reproducible without per-step duplication.

### Documentation updates

- `USER_GUIDE.md` — rewrote the expletive section around the three-tier
  detector; refreshed the available-ablations list to match
  `preprocessing/ablations/__init__.py`.
- `ADVANCED.md` — added tier counting, provenance fields, document
  boundary handling, and the optional coref confirmation workflow.
- `DEVELOPER_GUIDE.md` — documented the stateful-class pattern with
  `reset_file_state` / `get_file_tier_counts` hooks.
- `docs/data_processing.md` — replaced the stale script list with the
  current registry.

## Summary

Preprocessing documentation has been completely rewritten for clarity, usability, and maintainability. The changes prioritize human-centric design, remove unnecessary metadata, and consolidate redundant content.

## What Changed

### Files Updated

**Replaced with cleaner versions:**
- `README.md` - Reduced from 387 lines to ~90 lines
- `USER_GUIDE.md` - Rewritten with research context and practical workflows
- `DEVELOPER_GUIDE.md` - Simplified with clear examples and patterns
- `TESTING.md` - Focused on practical usage, removed outdated content

**Merged and simplified:**
- `ADVANCED.md` (NEW) - Consolidates:
  - `ADVANCED_USAGE.md` (deleted)
  - `PHASE4_ENHANCEMENTS.md` (deleted)
  - Focuses on when/why to use advanced features

**Removed:**
- `TEST_STATUS.md` - Redundant (status now in TESTING.md)
- `PHASE4_ENHANCEMENTS.md` - Merged into ADVANCED.md
- `ADVANCED_USAGE.md` - Merged into ADVANCED.md

### Old Files Preserved

Original files renamed with `_OLD` suffix for reference:
- `README_OLD.md`
- `USER_GUIDE_OLD.md`
- `DEVELOPER_GUIDE_OLD.md`
- `TESTING_OLD.md`

## Key Improvements

### 1. Removed Information Overload

**Before:**
```markdown
- **[User Guide](USER_GUIDE.md)** - Complete usage examples and workflows (600+ lines)
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Adding custom ablations (700+ lines)
### ✅ Maintainability
- 80% code reduction from legacy scripts
- Registry-based architecture
- Comprehensive test coverage (106 tests)
**Total: 72 tests passing** ✅
```

**After:**
```markdown
**New to preprocessing?** → [User Guide](USER_GUIDE.md)
**Need to customize?** → [Developer Guide](DEVELOPER_GUIDE.md)

**Maintainability**: Registry-based architecture makes adding new ablations straightforward.
```

### 2. User-Centric Structure

**Before**: Technical implementation details first
**After**: "What can I do?" and "Why does this matter?" first

Example from README.md:

**Old approach:**
```markdown
## Directory Structure
preprocessing/
├── __init__.py              # Public API
├── base.py                  # AblationPipeline class
...
```

**New approach:**
```markdown
## What This Does

Remove or modify specific linguistic features to create controlled experiments.
For example, remove all articles to test whether models can learn grammar
without "the", "a", or "an".

## Quick Start
[Immediate working example]
```

### 3. Removed Phase Branding

**Before**: "Phase 4 Enhancements", "Phase 5 Complete"
**After**: Feature-focused naming ("Performance Optimization", "Advanced Features")

### 4. Consolidated Content

**7 files → 5 files** (29% reduction)

**Before:**
```
README.md (387 lines)
USER_GUIDE.md (419 lines)
DEVELOPER_GUIDE.md (618 lines)
ADVANCED_USAGE.md (140 lines)
PHASE4_ENHANCEMENTS.md (270 lines)
TESTING.md (101 lines)
TEST_STATUS.md (92 lines)
Total: ~2,027 lines, 7 files
```

**After:**
```
README.md (~90 lines)
USER_GUIDE.md (~350 lines)
DEVELOPER_GUIDE.md (~400 lines)
ADVANCED.md (~380 lines)
TESTING.md (~400 lines)
Total: ~1,620 lines, 5 files
```

### 5. Improved Examples

**Before** (abstract):
```python
config = AblationConfig(
    type="remove_articles",  # Ablation type (registered name)
    input_path=Path("..."),  # Input corpus directory
    output_path=Path("..."), # Output directory
    seed=42                  # Random seed
)
```

**After** (concrete with context):
```python
# Remove articles from training corpus
config = AblationConfig(
    type="remove_articles",
    input_path="data/bnc_train/",
    output_path="data/bnc_no_articles/",
    seed=42
)
```

### 6. Better Navigation

**Before**: Circular references between 4 README files
**After**: Clear "Choose Your Path" in main README:

```markdown
**New to preprocessing?** → User Guide
**Need to customize?** → Developer Guide
**Large-scale processing?** → Advanced Usage
**Testing your changes?** → Testing Guide
```

### 7. Removed Status Markers

**Before**: ✅, 🚧, ▓▓▓░░░ everywhere
**After**: Clean prose, status only where relevant (test counts)

## File-by-File Changes

### README.md

**Improvements:**
- Reduced 387 → ~90 lines (77% reduction)
- Lead with "What This Does" not implementation
- Removed directory structure diagrams
- Removed changelog section
- Removed benefits comparison tables
- Added clear navigation paths
- Removed all emoji and status markers

**Key sections:**
- What This Does
- Quick Start
- Available Ablations (table)
- Choose Your Path (navigation)
- Example: Research Workflow
- Common Tasks

### USER_GUIDE.md

**Improvements:**
- Added research context ("Why this matters")
- Reorganized by user goals, not ablation types
- Removed redundant configuration listings
- Added concrete research examples
- Simplified performance tuning section
- Improved error handling examples

**Key sections:**
- Understanding Ablations (research context)
- Basic Usage
- Available Ablations (with use cases)
- Common Workflows
- Configuration Options
- Performance Tuning
- Error Handling
- Provenance Tracking
- Troubleshooting

### DEVELOPER_GUIDE.md

**Improvements:**
- Removed time estimates ("30 minutes")
- Removed numbered checklists
- Added conceptual overview
- Improved real-world examples
- Clearer common pitfalls section
- Better testing guidance

**Key sections:**
- Overview (registry pattern explained)
- Quick Start
- Ablation Function Interface
- Complete Example
- Real-World Examples
- Advanced Patterns
- Common Pitfalls
- Testing Your Ablation
- Debugging
- Performance Considerations

### ADVANCED.md (NEW)

**Consolidates:**
- ADVANCED_USAGE.md (coreference resolution)
- PHASE4_ENHANCEMENTS.md (performance, errors)

**Improvements:**
- Removed "Phase 4" branding
- Added "When You Need This" section
- Clearer performance comparison
- Better coreference explanation
- Production deployment patterns
- Cluster processing examples

**Key sections:**
- When You Need This
- Performance Optimization
- Coreference Resolution
- Production Deployment
- Cluster Processing
- Troubleshooting
- Best Practices Summary

### TESTING.md

**Improvements:**
- Focused on practical usage
- Removed NumPy incompatibility section (outdated)
- Removed detailed fixture documentation
- Added CI/CD integration example
- Better test structure examples

**Key sections:**
- Running Tests
- Test Status (current: 106/106 passing)
- Writing Tests for New Ablations
- Testing Best Practices
- Common Test Patterns
- Continuous Integration
- Debugging Failed Tests
- Coverage Goals

## Migration for Users

### Finding Information

**Old way:**
- README → link to USER_GUIDE → link to PHASE4_ENHANCEMENTS → find performance info
- 3+ clicks, unclear which doc to check

**New way:**
- README → "Large-scale processing?" → ADVANCED.md → Performance section
- 2 clicks, clear navigation

### Code Examples

All existing code examples still work. No breaking changes to the API, only documentation improvements.

### Old Documentation

Old files preserved with `_OLD` suffix for reference:
- `README_OLD.md`
- `USER_GUIDE_OLD.md`
- `DEVELOPER_GUIDE_OLD.md`
- `TESTING_OLD.md`

## What Was NOT Changed

- All code functionality (no API changes)
- All test files (tests still pass)
- Legacy preprocessing scripts (already archived)
- Core concepts and explanations (only clarified)

## Benefits

### For New Users

- Understand what preprocessing does in < 1 minute (README)
- Get first working example in < 5 minutes
- Clear path to relevant documentation

### For Existing Users

- Find information faster (fewer files, better navigation)
- Less visual clutter (no line counts, fewer status markers)
- More practical examples (research workflows)

### For Contributors

- Clearer developer guide with better examples
- Testing guide focused on practical patterns
- Less redundancy to maintain

### For Maintainers

- 20% fewer lines to maintain
- 29% fewer files to keep updated
- No phase branding to update
- No line counts to recalculate

## Next Steps

These changes can serve as a model for improving other documentation:
- model_foundry docs (12 files)
- Root-level docs (7+ files)
- Overall project structure

See [plans/comprehensive_documentation_improvement_plan.md](../../plans/comprehensive_documentation_improvement_plan.md) for the complete improvement plan.
