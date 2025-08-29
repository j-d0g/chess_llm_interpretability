#!/usr/bin/env python3
"""Extract probe training and testing results from log files for analysis."""

import re
import pandas as pd
import glob
from pathlib import Path

def extract_training_results():
    """Extract final training accuracies from error logs."""
    results = []
    
    # Pattern to match final accuracy lines
    pattern = r'layer (\d+), final acc: ([\d.]+)'
    
    # Find all training error logs
    log_files = glob.glob('logs/*_probes_*.err')
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract model info from filename
            filename = Path(log_file).stem
            
            # Parse model info from log content (look for model name)
            model_match = re.search(r'model_name.*tf_lens_([^,\s]+)', content)
            if model_match:
                model_name = model_match.group(1)
            else:
                # Fallback: extract from filename
                model_name = filename.split('_probes_')[0]
            
            # Find all final accuracy matches
            matches = re.findall(pattern, content)
            
            for layer_str, acc_str in matches:
                results.append({
                    'model': model_name,
                    'layer': int(layer_str),
                    'training_accuracy': float(acc_str),
                    'log_file': log_file
                })
                
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return pd.DataFrame(results)

def extract_test_results():
    """Extract test accuracies from test logs (once available)."""
    results = []
    
    # Pattern to match test accuracy lines
    pattern = r'Layer (\d+) test accuracy: ([\d.]+)'
    
    # Find all test output logs
    log_files = glob.glob('logs/test_*_probes_*.out') + glob.glob('logs/test_*_probes_*.err')
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract model info from filename
            filename = Path(log_file).stem
            model_name = filename.replace('test_', '').replace('_probes', '').split('_')[0:2]
            model_name = '_'.join(model_name)
            
            # Find all test accuracy matches
            matches = re.findall(pattern, content)
            
            for layer_str, acc_str in matches:
                results.append({
                    'model': model_name,
                    'layer': int(layer_str),
                    'test_accuracy': float(acc_str),
                    'log_file': log_file
                })
                
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    return pd.DataFrame(results)

def create_summary_table():
    """Create a comprehensive summary table."""
    print("Extracting training results...")
    train_df = extract_training_results()
    
    print("Extracting test results...")
    test_df = extract_test_results()
    
    if len(train_df) > 0:
        print(f"\nFound training results for {train_df['model'].nunique()} models:")
        print(train_df.groupby('model')['layer'].count().to_string())
        
        # Save training results
        train_df.to_csv('training_results.csv', index=False)
        print("\nSaved training_results.csv")
        
        # Show sample
        print("\nSample training results:")
        print(train_df.head(10).to_string(index=False))
    
    if len(test_df) > 0:
        print(f"\nFound test results for {test_df['model'].nunique()} models:")
        print(test_df.groupby('model')['layer'].count().to_string())
        
        # Save test results
        test_df.to_csv('test_results.csv', index=False)
        print("\nSaved test_results.csv")
        
        # Show sample
        print("\nSample test results:")
        print(test_df.head(10).to_string(index=False))
    
    # Merge if both available
    if len(train_df) > 0 and len(test_df) > 0:
        combined = pd.merge(train_df[['model', 'layer', 'training_accuracy']], 
                          test_df[['model', 'layer', 'test_accuracy']], 
                          on=['model', 'layer'], 
                          how='outer')
        combined.to_csv('combined_results.csv', index=False)
        print("\nSaved combined_results.csv")

if __name__ == "__main__":
    create_summary_table()