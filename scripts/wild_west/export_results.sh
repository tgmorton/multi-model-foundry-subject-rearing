#!/usr/bin/env bash
# Results Export Script
# Export evaluation results for R analysis and visualization

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Logging
log() {
    local level="$1"; shift
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    case "$level" in
        INFO)    echo -e "${BLUE}[$ts] INFO: $*${NC}" ;;
        WARN)    echo -e "${YELLOW}[$ts] WARN: $*${NC}" ;;
        ERROR)   echo -e "${RED}[$ts] ERROR: $*${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$ts] SUCCESS: $*${NC}" ;;
    esac
}

# Usage
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [experiments...]

Export evaluation results for R analysis and visualization.

Options:
  -o, --output <dir>        Output directory (default: evaluation/exports/)
  -f, --format <formats>    Export formats: csv,json,rds (default: csv,json)
  -a, --aggregate           Create cross-experiment aggregated file
  -p, --plots               Generate basic plots (requires Python with matplotlib)
  -c, --compress            Compress output files
  -v, --verbose             Verbose output
  -h, --help               Show this help

Arguments:
  experiments               Specific experiments to export (default: all)

Examples:
  # Export all experiments to CSV and JSON
  $0

  # Export specific experiments
  $0 exp0_baseline exp1_remove_expletives

  # Export with plots and compression
  $0 -p -c -o results_export/

  # Export only CSV format for all experiments
  $0 -f csv
EOF
}

# Find available experiments
find_experiments() {
    local results_dir="$PROJECT_DIR/evaluation/results"
    
    if [[ ! -d "$results_dir" ]]; then
        return 0
    fi
    
    find "$results_dir" -name "evaluation_results.jsonl" | while read -r results_file; do
        basename "$(dirname "$results_file")"
    done | sort
}

# Export single experiment
export_experiment() {
    local experiment="$1"
    local output_dir="$2"
    local formats="$3"
    local verbose="$4"
    
    local results_file="$PROJECT_DIR/evaluation/results/$experiment/evaluation_results.jsonl"
    
    if [[ ! -f "$results_file" ]]; then
        log WARN "No results file found for $experiment"
        return 1
    fi
    
    log INFO "Exporting $experiment..."
    
    # Create experiment output directory
    local exp_output_dir="$output_dir/$experiment"
    mkdir -p "$exp_output_dir"
    
    # Use Python to export
    local export_script="/tmp/export_experiment_$$.py"
    cat > "$export_script" <<EOF
import sys
import json
import pandas as pd
import os
from pathlib import Path

sys.path.insert(0, '$PROJECT_DIR')

try:
    from evaluation.result_aggregator import ResultAggregator
    
    # Initialize aggregator
    aggregator = ResultAggregator('$exp_output_dir')
    
    # Load and process results
    results = aggregator.load_evaluation_results('$results_file')
    
    if not results:
        print("No results found")
        sys.exit(1)
    
    print(f"Processing {len(results)} checkpoints...")
    
    # Export formats
    formats = '$formats'.split(',')
    
    exported_files = {}
    
    if 'csv' in formats:
        # Extract metrics for CSV
        metrics_df = aggregator.extract_scalar_metrics(results)
        
        # Save main metrics
        metrics_file = Path('$exp_output_dir') / '${experiment}_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        exported_files['metrics_csv'] = str(metrics_file)
        
        # Save learning curves
        curves_df = aggregator.create_learning_curves(metrics_df)
        curves_file = Path('$exp_output_dir') / '${experiment}_learning_curves.csv'
        curves_df.to_csv(curves_file, index=False)
        exported_files['curves_csv'] = str(curves_file)
        
        # Create wide format if possible
        try:
            pivot_cols = ['checkpoint']
            if 'epoch' in metrics_df.columns:
                pivot_cols.append('epoch')
            if 'step' in metrics_df.columns:
                pivot_cols.append('step')
                
            wide_df = metrics_df.pivot_table(
                index=pivot_cols,
                columns='metric',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            wide_file = Path('$exp_output_dir') / '${experiment}_wide.csv'
            wide_df.to_csv(wide_file, index=False)
            exported_files['wide_csv'] = str(wide_file)
            
        except Exception as e:
            if '$verbose' == 'true':
                print(f"Could not create wide format: {e}")
    
    if 'json' in formats:
        # Save processed JSON
        summary = aggregator.generate_summary_report(results)
        
        summary_file = Path('$exp_output_dir') / '${experiment}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        exported_files['summary_json'] = str(summary_file)
        
        # Save raw results (pretty printed)
        raw_file = Path('$exp_output_dir') / '${experiment}_raw.json'
        with open(raw_file, 'w') as f:
            json.dump(results, f, indent=2)
        exported_files['raw_json'] = str(raw_file)
    
    # Print exported files
    for file_type, filepath in exported_files.items():
        print(f"  {file_type}: {filepath}")
    
    print(f"Exported {experiment} successfully")

except Exception as e:
    print(f"Error exporting {experiment}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    # Run export script
    cd "$PROJECT_DIR"
    if python3 "$export_script"; then
        log SUCCESS "Exported $experiment"
    else
        log ERROR "Failed to export $experiment"
        return 1
    fi
    
    # Cleanup
    rm -f "$export_script"
}

# Create aggregated cross-experiment file
create_aggregated_export() {
    local experiments=("$@")
    local output_dir="${experiments[-1]}"
    unset 'experiments[-1]'
    
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log WARN "No experiments to aggregate"
        return 0
    fi
    
    log INFO "Creating cross-experiment aggregated file..."
    
    local aggregate_script="/tmp/aggregate_experiments_$$.py"
    cat > "$aggregate_script" <<EOF
import sys
import pandas as pd
import json
from pathlib import Path

sys.path.insert(0, '$PROJECT_DIR')

try:
    from evaluation.result_aggregator import ResultAggregator
    
    experiments = [${experiments[*]@Q}]
    results_base = '$PROJECT_DIR/evaluation/results'
    
    all_data = []
    
    for exp in experiments:
        results_file = Path(results_base) / exp / 'evaluation_results.jsonl'
        
        if not results_file.exists():
            print(f"Warning: No results for {exp}")
            continue
        
        # Load and process
        aggregator = ResultAggregator('$output_dir')
        results = aggregator.load_evaluation_results(str(results_file))
        
        if results:
            metrics_df = aggregator.extract_scalar_metrics(results)
            metrics_df['experiment'] = exp
            all_data.append(metrics_df)
    
    if not all_data:
        print("No data to aggregate")
        sys.exit(1)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined file
    output_file = Path('$output_dir') / 'cross_experiment_comparison.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Create summary stats
    summary_stats = combined_df.groupby(['experiment', 'metric'])['value'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    summary_file = Path('$output_dir') / 'cross_experiment_summary.csv'
    summary_stats.to_csv(summary_file, index=False)
    
    print(f"Aggregated {len(experiments)} experiments")
    print(f"  Combined data: {output_file}")
    print(f"  Summary stats: {summary_file}")
    print(f"  Total rows: {len(combined_df)}")
    
except Exception as e:
    print(f"Error creating aggregated export: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    cd "$PROJECT_DIR"
    if python3 "$aggregate_script"; then
        log SUCCESS "Created aggregated export"
    else
        log ERROR "Failed to create aggregated export"
        return 1
    fi
    
    rm -f "$aggregate_script"
}

# Generate plots
generate_plots() {
    local output_dir="$1"
    local experiments=("${@:2}")
    
    log INFO "Generating plots..."
    
    local plot_script="/tmp/generate_plots_$$.py"
    cat > "$plot_script" <<EOF
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    output_dir = Path('$output_dir')
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load aggregated data if available
    agg_file = output_dir / 'cross_experiment_comparison.csv'
    
    if agg_file.exists():
        df = pd.read_csv(agg_file)
        
        # Learning curves plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Curves Across Experiments', fontsize=16)
        
        metrics = ['perplexity', 'blimp_overall', 'null_subject_overt_pref', 'null_subject_surprisal_diff']
        titles = ['Perplexity', 'BLIMP Accuracy', 'Null-Subject Overt Preference', 'Null-Subject Surprisal Difference']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            metric_data = df[df['metric'] == metric]
            
            if not metric_data.empty:
                # Sort by epoch/step if available
                if 'epoch' in metric_data.columns:
                    metric_data = metric_data.sort_values(['experiment', 'epoch'])
                    x_col = 'epoch'
                elif 'step' in metric_data.columns:
                    metric_data = metric_data.sort_values(['experiment', 'step'])  
                    x_col = 'step'
                else:
                    x_col = None
                
                if x_col:
                    for exp in metric_data['experiment'].unique():
                        exp_data = metric_data[metric_data['experiment'] == exp]
                        ax.plot(exp_data[x_col], exp_data['value'], marker='o', label=exp)
                
                ax.set_title(title)
                ax.set_xlabel(x_col if x_col else 'Checkpoint')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Final performance comparison
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get final values for each experiment
        final_values = []
        for exp in df['experiment'].unique():
            exp_data = df[df['experiment'] == exp]
            
            # Get final checkpoint data
            if 'epoch' in exp_data.columns:
                final_data = exp_data.loc[exp_data.groupby('metric')['epoch'].idxmax()]
            elif 'step' in exp_data.columns:
                final_data = exp_data.loc[exp_data.groupby('metric')['step'].idxmax()]
            else:
                final_data = exp_data.groupby('metric').tail(1)
            
            for _, row in final_data.iterrows():
                final_values.append({
                    'experiment': exp,
                    'metric': row['metric'],
                    'value': row['value']
                })
        
        if final_values:
            final_df = pd.DataFrame(final_values)
            
            # Create heatmap of final performance
            pivot_df = final_df.pivot(index='experiment', columns='metric', values='value')
            
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
            ax.set_title('Final Performance Comparison')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'final_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {plots_dir}/")
        print("  learning_curves.png")
        print("  final_comparison.png")
    
    else:
        print("No aggregated data found for plotting")

except ImportError:
    print("Matplotlib/Seaborn not available - skipping plots")
except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc()
EOF
    
    cd "$PROJECT_DIR"
    python3 "$plot_script"
    rm -f "$plot_script"
}

# Compress output files
compress_output() {
    local output_dir="$1"
    
    log INFO "Compressing output files..."
    
    cd "$output_dir"
    
    # Create tar.gz archive
    local archive_name="evaluation_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    if tar -czf "$archive_name" *.csv *.json */; then
        local archive_size
        archive_size=$(ls -lh "$archive_name" | awk '{print $5}')
        log SUCCESS "Created compressed archive: $archive_name ($archive_size)"
        
        # Optionally remove uncompressed files
        read -p "Remove uncompressed files? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find . -name "*.csv" -o -name "*.json" | head -10 | xargs rm -f
            log INFO "Removed uncompressed files"
        fi
    else
        log ERROR "Failed to create archive"
        return 1
    fi
}

# Main function
main() {
    local output_dir="$PROJECT_DIR/evaluation/exports"
    local formats="csv,json"
    local experiments=()
    local aggregate=false
    local plots=false
    local compress=false
    local verbose=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output)
                output_dir="$2"
                shift 2
                ;;
            -f|--format)
                formats="$2"
                shift 2
                ;;
            -a|--aggregate)
                aggregate=true
                shift
                ;;
            -p|--plots)
                plots=true
                shift
                ;;
            -c|--compress)
                compress=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                log ERROR "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                experiments+=("$1")
                shift
                ;;
        esac
    done
    
    # Find experiments if none specified
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log INFO "Finding available experiments..."
        while IFS= read -r exp; do
            experiments+=("$exp")
        done < <(find_experiments)
    fi
    
    if [[ ${#experiments[@]} -eq 0 ]]; then
        log ERROR "No experiments found to export"
        exit 1
    fi
    
    log INFO "Exporting ${#experiments[@]} experiments: ${experiments[*]}"
    log INFO "Output directory: $output_dir"
    log INFO "Formats: $formats"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Export each experiment
    local exported_count=0
    for experiment in "${experiments[@]}"; do
        if export_experiment "$experiment" "$output_dir" "$formats" "$verbose"; then
            ((exported_count++))
        fi
    done
    
    log SUCCESS "Exported $exported_count experiments"
    
    # Create aggregated file if requested
    if [[ "$aggregate" == "true" ]] && [[ $exported_count -gt 1 ]]; then
        create_aggregated_export "${experiments[@]}" "$output_dir"
    fi
    
    # Generate plots if requested
    if [[ "$plots" == "true" ]]; then
        generate_plots "$output_dir" "${experiments[@]}"
    fi
    
    # Compress output if requested
    if [[ "$compress" == "true" ]]; then
        compress_output "$output_dir"
    fi
    
    log SUCCESS "Export completed! Files saved to: $output_dir"
    
    # Show summary
    echo
    echo "=== Export Summary ==="
    find "$output_dir" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.png" -o -name "*.tar.gz" \) | \
    while read -r file; do
        local size
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  $(basename "$file"): $size"
    done
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi