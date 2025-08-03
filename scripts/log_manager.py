#!/usr/bin/env python3
"""
Log Manager - A simple tool to view and manage experiment logs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add model_foundry to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model_foundry"))
from logging_utils import get_latest_log, list_experiment_logs, cleanup_empty_logs


def main():
    parser = argparse.ArgumentParser(description="Manage and view experiment logs")
    parser.add_argument("action", choices=["list", "view", "clean", "latest"], 
                       help="Action to perform")
    parser.add_argument("--experiment", "-e", help="Experiment name")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--lines", "-n", type=int, default=50, 
                       help="Number of lines to show (for view action)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        if args.experiment:
            # List logs for specific experiment
            logs = list_experiment_logs(args.experiment, args.log_dir)
            if not logs:
                print(f"No logs found for experiment '{args.experiment}'")
                return
            
            print(f"\nLogs for experiment '{args.experiment}':")
            print("-" * 80)
            for filename, timestamp, size in logs:
                size_str = f"{size:,} bytes" if size > 0 else "empty"
                print(f"{timestamp} | {filename} | {size_str}")
        else:
            # List all experiments
            log_dir = Path(args.log_dir)
            if not log_dir.exists():
                print(f"Log directory '{args.log_dir}' does not exist")
                return
            
            experiments = [d.name for d in log_dir.iterdir() if d.is_dir()]
            if not experiments:
                print("No experiments found")
                return
            
            print("\nAvailable experiments:")
            print("-" * 40)
            for exp in sorted(experiments):
                logs = list_experiment_logs(exp, args.log_dir, max_files=1)
                latest = logs[0] if logs else None
                if latest:
                    print(f"{exp}: {latest[1]} ({latest[2]:,} bytes)")
                else:
                    print(f"{exp}: No logs")
    
    elif args.action == "view":
        if not args.experiment:
            print("Error: --experiment is required for 'view' action")
            return
        
        latest_log = get_latest_log(args.experiment, args.log_dir)
        if not latest_log:
            print(f"No logs found for experiment '{args.experiment}'")
            return
        
        print(f"\nViewing latest log for '{args.experiment}': {latest_log.name}")
        print("-" * 80)
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if len(lines) <= args.lines:
                    print(''.join(lines))
                else:
                    print(f"... showing last {args.lines} lines ...")
                    print(''.join(lines[-args.lines:]))
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    elif args.action == "latest":
        if not args.experiment:
            print("Error: --experiment is required for 'latest' action")
            return
        
        latest_log = get_latest_log(args.experiment, args.log_dir)
        if latest_log:
            stat = latest_log.stat()
            print(f"Latest log for '{args.experiment}':")
            print(f"  File: {latest_log}")
            print(f"  Size: {stat.st_size:,} bytes")
            print(f"  Modified: {datetime.fromtimestamp(stat.st_mtime)}")
        else:
            print(f"No logs found for experiment '{args.experiment}'")
    
    elif args.action == "clean":
        print("Cleaning up empty log files...")
        cleanup_empty_logs(args.log_dir)
        print("Done!")


if __name__ == "__main__":
    main() 