# Legacy Preprocessing Scripts

These are the original preprocessing scripts that have been replaced by the unified pipeline system.

## Status

**⚠️ DEPRECATED** - These scripts are archived and should not be used for new work.

Use the new unified system instead:
```python
from preprocessing.config import AblationConfig
from preprocessing.base import AblationPipeline

config = AblationConfig(
    type="remove_articles",  # or any other ablation
    input_path="data/raw/",
    output_path="data/processed/",
    seed=42
)

pipeline = AblationPipeline(config)
manifest = pipeline.process_corpus()
```

## Files

- `remove_articles.py` - Original article removal script (365 lines)
- `remove_expletives.py` - Original expletive removal script (410 lines)
- `impoverish_determiners.py` - Original determiner impoverishment script (373 lines)
- `lemmatize_verbs.py` - Original verb lemmatization script (371 lines)
- `remove_subject_pronominals.py` - Original subject pronoun removal script (~350 lines)

**Total**: ~1,869 lines of duplicated code

## New System Benefits

The new unified system in `preprocessing/` provides:

- ✅ **80% code reduction**: ~1,500 lines eliminated through refactoring
- ✅ **Type safety**: Pydantic configuration models
- ✅ **Reproducibility**: Automatic seed setting and provenance tracking
- ✅ **Error handling**: Graceful degradation, detailed logging
- ✅ **Performance**: 30-40% faster with optimizations
- ✅ **Testability**: 106 tests ensure correctness
- ✅ **Maintainability**: Registry-based architecture

## Migration

See the [Preprocessing Documentation](../../docs/preprocessing/README.md) for:
- Migration guide from legacy scripts
- Complete API documentation
- Usage examples
- Developer guide for custom ablations

## Why Keep These?

These files are archived for:
1. **Reference**: Understanding original implementation decisions
2. **Validation**: Regression testing against original behavior
3. **Documentation**: Historical record of the refactoring process

## History

- **Phases 1-2** (Oct 2025): Base infrastructure and first ablation refactored
- **Phase 3** (Oct 2025): All 5 ablations migrated to new system
- **Phase 4** (Oct 2025): Enhanced error handling and performance
- **Phase 5** (Oct 2025): Complete documentation, scripts archived

For the complete refactoring plan, see `plans/preprocessing_refactor_plan.md`.
