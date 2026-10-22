"""
Test script for the new checkpoint scheduling system.
"""

import os
import tempfile
import yaml
from pathlib import Path
import sys

# Add the model_foundry package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.config import ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig, TrainingConfig, LoggingConfig
from scripts.generate_checkpoint_schedule import (
    CheckpointGenerationConfig,
    generate_first_epoch_checkpoints,
    generate_subsequent_epoch_checkpoints,
    calculate_steps_per_epoch,
    generate_checkpoint_schedule
)


def create_test_config():
    """Create a test configuration."""
    return ExperimentConfig(
        experiment_name="test_new_checkpoint_experiment",
        data=DataConfig(
            source_corpus="test_data",
            training_corpus="test_data",
            batch_size=4,
            max_sequence_length=128
        ),
        tokenizer=TokenizerConfig(
            output_dir="test_tokenizer",
            vocab_size=1000
        ),
        model=ModelConfig(
            layers=2,
            embedding_size=64,
            hidden_size=64,
            intermediate_hidden_size=128,
            attention_heads=4,
            activation_function="GELU",
            dropout=0.1,
            attention_dropout=0.1
        ),
        training=TrainingConfig(
            output_dir="test_models",
            learning_rate=0.0001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6,
            warmup_steps=100,
            train_steps=1000,
            epochs=3,
            auto_generate_checkpoints=False,
            first_epoch_checkpoints=5,
            subsequent_epochs_spacing="log",
            log_base=2,
            linear_interval=None,
            min_checkpoint_interval=50
        ),
        logging=LoggingConfig(
            use_wandb=False
        ),
        random_seed=42
    )


def test_checkpoint_generation_config():
    """Test the new CheckpointGenerationConfig class."""
    config = CheckpointGenerationConfig(
        first_epoch_checkpoints=10,
        subsequent_epochs_spacing="log",
        log_base=2,
        linear_interval=None,
        min_interval=100
    )
    
    assert config.first_epoch_checkpoints == 10
    assert config.subsequent_epochs_spacing == "log"
    assert config.log_base == 2
    assert config.linear_interval is None
    assert config.min_interval == 100
    
    # Test validation
    try:
        CheckpointGenerationConfig(subsequent_epochs_spacing="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_generate_first_epoch_checkpoints():
    """Test first epoch checkpoint generation."""
    # Test with 5 checkpoints in 100 steps
    checkpoints = generate_first_epoch_checkpoints(100, 5)
    
    assert len(checkpoints) == 5
    assert checkpoints[0] == 0  # First checkpoint
    assert checkpoints[-1] == 100  # Last checkpoint
    
    # Check that checkpoints are evenly distributed
    intervals = [checkpoints[i+1] - checkpoints[i] for i in range(len(checkpoints)-1)]
    assert all(interval > 0 for interval in intervals)
    
    # Test edge cases
    assert generate_first_epoch_checkpoints(100, 0) == []
    assert generate_first_epoch_checkpoints(100, 1) == [100]


def test_generate_subsequent_epoch_checkpoints():
    """Test subsequent epoch checkpoint generation."""
    epoch_start = 100
    epoch_end = 200

    # Test linear spacing (min-per-epoch fill disabled)
    linear_checkpoints = generate_subsequent_epoch_checkpoints(
        epoch_start, epoch_end, "linear", linear_interval=20,
        min_checkpoints_per_epoch=0
    )
    assert linear_checkpoints == [120, 140, 160, 180]

    # Test log spacing (min-per-epoch fill disabled): each step is the
    # previous one times log_base, starting from epoch_start + log_base
    log_checkpoints = generate_subsequent_epoch_checkpoints(
        epoch_start, 1000, "log", log_base=2, min_checkpoints_per_epoch=0
    )
    assert log_checkpoints == [102, 204, 408, 816]

    # With the default min_checkpoints_per_epoch=5, log spacing over a short
    # epoch yields only [102], so it is replaced by 5 evenly spaced steps
    filled_checkpoints = generate_subsequent_epoch_checkpoints(
        epoch_start, epoch_end, "log", log_base=2
    )
    assert filled_checkpoints == [116, 132, 148, 164, 180]
    for checkpoint in filled_checkpoints:
        assert epoch_start < checkpoint < epoch_end


def test_calculate_steps_per_epoch():
    """Test steps per epoch calculation."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    # Test fallback calculation
    steps = calculate_steps_per_epoch(config, base_dir)
    assert steps > 0
    assert steps == config.training.train_steps // config.training.epochs


def test_generate_checkpoint_schedule():
    """Test the complete checkpoint schedule generation."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    generation_config = CheckpointGenerationConfig(
        first_epoch_checkpoints=5,
        subsequent_epochs_spacing="log",
        log_base=2,
        linear_interval=None,
        min_interval=50
    )
    
    schedule = generate_checkpoint_schedule(config, base_dir, generation_config)
    
    # Should have generated a schedule
    assert len(schedule) > 0
    
    # All steps should be within training range
    for step in schedule:
        assert 0 <= step <= config.training.train_steps

    # Should include step 0 (the initialization checkpoint)
    assert schedule[0] == 0
    
    # Should be sorted
    assert schedule == sorted(schedule)
    
    # Should include final step
    assert config.training.train_steps in schedule
    
    # Should include epoch boundaries
    steps_per_epoch = config.training.train_steps // config.training.epochs
    for epoch in range(1, config.training.epochs + 1):
        epoch_step = epoch * steps_per_epoch
        if epoch_step <= config.training.train_steps:
            assert epoch_step in schedule


def test_linear_spacing():
    """Test linear spacing configuration."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    generation_config = CheckpointGenerationConfig(
        first_epoch_checkpoints=3,
        subsequent_epochs_spacing="linear",
        log_base=2,
        linear_interval=50,
        min_interval=25
    )
    
    schedule = generate_checkpoint_schedule(config, base_dir, generation_config)
    
    # Should have generated a schedule
    assert len(schedule) > 0
    
    # Check that linear intervals are respected where possible
    steps_per_epoch = config.training.train_steps // config.training.epochs
    
    # Check second epoch (should have linear spacing)
    second_epoch_start = steps_per_epoch
    second_epoch_end = 2 * steps_per_epoch
    
    second_epoch_checkpoints = [s for s in schedule if second_epoch_start < s <= second_epoch_end]
    
    if len(second_epoch_checkpoints) > 1:
        intervals = [second_epoch_checkpoints[i+1] - second_epoch_checkpoints[i] 
                    for i in range(len(second_epoch_checkpoints)-1)]
        # Should be approximately linear_interval
        for interval in intervals:
            assert abs(interval - generation_config.linear_interval) <= generation_config.linear_interval


def test_config_integration():
    """Test integration with the configuration system."""
    config = create_test_config()
    
    # Test that new checkpoint parameters are accessible
    assert config.training.first_epoch_checkpoints == 5
    assert config.training.subsequent_epochs_spacing == "log"
    assert config.training.log_base == 2
    assert config.training.linear_interval is None
    assert config.training.min_checkpoint_interval == 50
    
    # Test that we can set checkpoint schedule
    test_schedule = [100, 200, 300, 400, 500]
    config.training.checkpoint_schedule = test_schedule
    assert config.training.checkpoint_schedule == test_schedule


if __name__ == "__main__":
    print("Running new checkpoint scheduling tests...")
    
    test_checkpoint_generation_config()
    print("✓ CheckpointGenerationConfig test passed")
    
    test_generate_first_epoch_checkpoints()
    print("✓ First epoch checkpoint generation test passed")
    
    test_generate_subsequent_epoch_checkpoints()
    print("✓ Subsequent epoch checkpoint generation test passed")
    
    test_calculate_steps_per_epoch()
    print("✓ Steps per epoch calculation test passed")
    
    test_generate_checkpoint_schedule()
    print("✓ Complete checkpoint schedule generation test passed")
    
    test_linear_spacing()
    print("✓ Linear spacing test passed")
    
    test_config_integration()
    print("✓ Configuration integration test passed")
    
    print("\nAll new checkpoint scheduling tests passed! ✓") 