"""
Simple unit tests for evaluation pipeline.
Tests basic functionality without requiring full model checkpoints.
"""

import tempfile
import json
import csv
from pathlib import Path
import pytest

# Note: These tests focus on basic functionality
# Full integration tests would require actual model checkpoints


def test_config_loading():
    """Test that evaluation config can be loaded and validated."""
    from evaluation_runner import EvaluationConfig
    
    # Create sample config
    config_data = {
        "model_checkpoint_dir": "/tmp/models",
        "tokenizer_path": "/tmp/tokenizer",
        "test_corpus": "/tmp/test",
        "output_dir": "/tmp/results"
    }
    
    # Test config creation
    config = EvaluationConfig(**config_data)
    assert config.model_checkpoint_dir == "/tmp/models"
    assert config.batch_size == 32  # Default value
    assert config.run_perplexity is True  # Default value


def test_blimp_file_loading():
    """Test BLIMP file loading functionality."""
    from blimp_evaluator import BLIMPEvaluator
    
    # Create temporary BLIMP file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write sample BLIMP data
        sample_data = [
            {
                "sentence_good": "The cat sat on the mat.",
                "sentence_bad": "The cat on the mat sat.",
                "field": "syntax",
                "linguistics_term": "word_order",
                "UID": "test_uid",
                "pairID": "1"
            },
            {
                "sentence_good": "She likes apples.",
                "sentence_bad": "Her likes apples.",
                "field": "morphology", 
                "linguistics_term": "case",
                "UID": "test_uid2",
                "pairID": "2"
            }
        ]
        
        for item in sample_data:
            json.dump(item, f)
            f.write('\n')
        
        temp_file = f.name
    
    # Test loading (without surprisal calculator)
    try:
        # Create mock evaluator
        evaluator = BLIMPEvaluator(None)  # No calculator for this test
        stimuli = evaluator.load_blimp_file(temp_file)
        
        assert len(stimuli) == 2
        assert stimuli[0]['sentence_good'] == "The cat sat on the mat."
        assert stimuli[1]['linguistics_term'] == "case"
        
    finally:
        # Clean up
        Path(temp_file).unlink()


def test_null_subject_file_loading():
    """Test null-subject stimuli file loading."""
    from null_subject_evaluator import NullSubjectEvaluator
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['item', 'item_group', 'pronoun_status', 'c_english', 'target', 'hotspot_english'])
        
        # Write sample data
        writer.writerow([1, '1a_3rdSG', 1, 'Mary won the award.', 'She shows pride.', 'shows'])
        writer.writerow([1, '1a_3rdSG', 0, 'Mary won the award.', 'Shows pride.', 'shows'])
        writer.writerow([2, '1a_3rdSG', 1, 'The dog barked.', 'It scared cats.', 'scared'])
        writer.writerow([2, '1a_3rdSG', 0, 'The dog barked.', 'Scared cats.', 'scared'])
        
        temp_file = f.name
    
    try:
        # Test loading
        evaluator = NullSubjectEvaluator(None)  # No calculator for this test
        df = evaluator.load_stimuli_file(temp_file)
        
        assert len(df) == 4
        assert 'item' in df.columns
        assert 'pronoun_status' in df.columns
        assert df['item'].nunique() == 2  # Two different items
        
    finally:
        # Clean up
        Path(temp_file).unlink()


def test_result_aggregation():
    """Test result aggregation functionality."""
    from result_aggregator import ResultAggregator
    
    # Create sample results
    sample_results = [
        {
            "checkpoint": "epoch_1",
            "timestamp": "2025-01-01T12:00:00",
            "perplexity": {"perplexity": 50.0, "num_sequences": 100},
            "blimp": {"overall_accuracy": 0.6, "total_stimuli": 1000},
            "null_subject": {"overt_preference_rate": 0.8, "total_pairs": 200}
        },
        {
            "checkpoint": "epoch_5", 
            "timestamp": "2025-01-01T12:30:00",
            "perplexity": {"perplexity": 45.0, "num_sequences": 100},
            "blimp": {"overall_accuracy": 0.65, "total_stimuli": 1000},
            "null_subject": {"overt_preference_rate": 0.75, "total_pairs": 200}
        }
    ]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        aggregator = ResultAggregator(temp_dir)
        
        # Test scalar metrics extraction
        metrics_df = aggregator.extract_scalar_metrics(sample_results)
        
        assert len(metrics_df) > 0
        assert 'checkpoint' in metrics_df.columns
        assert 'metric' in metrics_df.columns
        assert 'value' in metrics_df.columns
        
        # Check that we have expected metrics
        metric_names = set(metrics_df['metric'].values)
        expected_metrics = {'perplexity', 'blimp_overall', 'null_subject_overt_pref'}
        assert expected_metrics.issubset(metric_names)
        
        # Test summary generation
        summary = aggregator.generate_summary_report(sample_results)
        assert summary['total_checkpoints'] == 2
        assert 'final_perplexity' in summary
        assert summary['final_perplexity'] == 45.0


def test_checkpoint_finding():
    """Test checkpoint discovery functionality."""
    from model_loader import ModelLoader
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock checkpoint directories
        (temp_path / "checkpoint-100").mkdir()
        (temp_path / "checkpoint-500").mkdir() 
        (temp_path / "epoch_1").mkdir()
        (temp_path / "epoch_10").mkdir()
        (temp_path / "other_file.txt").touch()  # Should be ignored
        
        # Test checkpoint finding
        loader = ModelLoader()
        checkpoints = loader.find_checkpoints(str(temp_path))
        
        # Should find 4 checkpoint directories
        assert len(checkpoints) == 4
        
        # Should be sorted properly (checkpoint-100 before checkpoint-500)
        checkpoint_names = [p.name for p in checkpoints]
        assert 'checkpoint-100' in checkpoint_names
        assert 'checkpoint-500' in checkpoint_names
        assert 'epoch_1' in checkpoint_names
        assert 'epoch_10' in checkpoint_names


def test_memory_usage_calculation():
    """Test memory usage calculation utilities."""
    from model_loader import get_model_memory_usage
    import torch
    
    # Create a small test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 5)
        
        def forward(self, x):
            return self.layer(x)
    
    model = TestModel()
    stats = get_model_memory_usage(model)
    
    assert 'param_memory_mb' in stats
    assert 'total_memory_mb' in stats
    assert stats['param_memory_mb'] > 0
    assert stats['total_memory_mb'] >= stats['param_memory_mb']


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("Running basic functionality tests...")
    
    try:
        test_config_loading()
        print("✓ Config loading test passed")
        
        test_blimp_file_loading() 
        print("✓ BLIMP file loading test passed")
        
        test_null_subject_file_loading()
        print("✓ Null-subject file loading test passed")
        
        test_result_aggregation()
        print("✓ Result aggregation test passed")
        
        test_checkpoint_finding()
        print("✓ Checkpoint finding test passed")
        
        test_memory_usage_calculation()
        print("✓ Memory usage calculation test passed")
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise