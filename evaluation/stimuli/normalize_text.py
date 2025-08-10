import pandas as pd
import re

def normalize_text_column(text):
    """Normalize text: ensure it starts with a space and is lowercase"""
    if pd.isna(text) or text == '':
        return text
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Ensure it starts with a space
    if not text.startswith(' '):
        text = ' ' + text
    
    return text

def normalize_csv_file(input_file, output_file=None):
    """Normalize the context, target, and hotspot columns in a CSV file"""
    
    # If no output file specified, overwrite the input
    if output_file is None:
        output_file = input_file
    
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file, index_col=0)
    
    print(f"Original file shape: {df.shape}")
    print(f"Columns to normalize: c_english, target, hotspot_english")
    
    # Show some examples before normalization
    print("\nBefore normalization (first 5 rows):")
    for col in ['c_english', 'target', 'hotspot_english']:
        if col in df.columns:
            print(f"{col}:")
            for i, val in enumerate(df[col].head(5)):
                print(f"  [{i}]: '{val}'")
    
    # Normalize the specified columns
    columns_to_normalize = ['c_english', 'target', 'hotspot_english']
    
    for col in columns_to_normalize:
        if col in df.columns:
            print(f"\nNormalizing column: {col}")
            df[col] = df[col].apply(normalize_text_column)
        else:
            print(f"Warning: Column '{col}' not found in the dataframe")
    
    # Show some examples after normalization
    print("\nAfter normalization (first 5 rows):")
    for col in ['c_english', 'target', 'hotspot_english']:
        if col in df.columns:
            print(f"{col}:")
            for i, val in enumerate(df[col].head(5)):
                print(f"  [{i}]: '{val}'")
    
    # Save the normalized file
    print(f"\nSaving normalized file: {output_file}")
    df.to_csv(output_file, index_label='index')
    
    print("Normalization complete!")
    return df

if __name__ == "__main__":
    # Check which file to normalize
    import os
    
    # First try the original file
    if os.path.exists('long_format/master_stimuli_with_forms.csv'):
        print("Found master_stimuli_with_forms.csv")
        normalize_csv_file('long_format/master_stimuli_with_forms.csv')
    else:
        print("master_stimuli_with_forms.csv not found")
    
    # Also check for the transformed file when it's ready
    if os.path.exists('long_format/master_stimuli_transformed.csv'):
        print("\nFound master_stimuli_transformed.csv")
        response = input("Also normalize the transformed file? (y/n): ")
        if response.lower().startswith('y'):
            normalize_csv_file('long_format/master_stimuli_transformed.csv')
    else:
        print("master_stimuli_transformed.csv not found yet (transformation still running)")
    
    print("\nDone!")