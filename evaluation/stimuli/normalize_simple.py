import pandas as pd
import os

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

def normalize_file(file_path):
    """Normalize a single file"""
    print(f"Normalizing: {file_path}")
    try:
        # Try reading with UTF-8 first
        df = pd.read_csv(file_path, index_col=0, encoding='utf-8')
    except UnicodeDecodeError:
        # If that fails, try with latin-1 encoding
        print(f"  UTF-8 failed, trying latin-1 encoding...")
        try:
            df = pd.read_csv(file_path, index_col=0, encoding='latin-1')
        except:
            # If that also fails, try with errors='ignore'
            print(f"  Latin-1 failed, reading with errors ignored...")
            df = pd.read_csv(file_path, index_col=0, encoding='utf-8', errors='ignore')
    
    # Normalize the specified columns
    columns_to_normalize = ['c_english', 'target', 'hotspot_english']
    
    for col in columns_to_normalize:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text_column)
    
    # Save back to the same file with UTF-8 encoding
    df.to_csv(file_path, index_label='index', encoding='utf-8')
    print(f"Completed: {file_path}")

# Normalize all relevant files
files_to_check = [
    'long_format/master_stimuli_with_forms.csv',
    'long_format/master_stimuli_transformed.csv'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        normalize_file(file_path)
    else:
        print(f"File not found: {file_path}")

print("All normalization complete!")