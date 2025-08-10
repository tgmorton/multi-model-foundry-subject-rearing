import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os

# DeepSeek API configuration
API_KEY = "sk-9e8d7d2603d9449e84d83402c5736603"
API_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

MANIPULATION_INSTRUCTIONS = {
    'complex_long': 'Rewrite the CONTEXT sentence to include more descriptive, longer noun phrases (NPs). For example, "the dog" could become "the large brown dog with the red collar". Do not change the TARGET sentence.',
    'complex_emb': 'Rewrite the CONTEXT sentence by adding an embedded relative clause. For example, "the dog barked" could become "the dog that lived down the street barked". Do not change the TARGET sentence.',
    'target_negation': "Rewrite the TARGET sentence to be negative. For example, \"She thinks the ending is perfect\" becomes \"She doesn't think the ending is perfect\". Do not change the CONTEXT sentence.",
    'context_negation': "Rewrite the CONTEXT sentence to be negative. For example, \"Anna finished the book\" becomes \"Anna didn't finish the book\". Do not change the TARGET sentence.",
    'both_negation': 'Rewrite BOTH the CONTEXT and TARGET sentences to be negative.',
}

def call_deepseek_api(context, target, manipulation):
    """Call DeepSeek API to transform stimuli"""
    
    instruction = MANIPULATION_INSTRUCTIONS[manipulation]
    
    prompt = f"""
You are a linguistics research assistant. Your task is to manipulate sentences according to specific instructions and extract key linguistic features.

**Instructions:**
1. You will be given a context sentence and a target sentence.
2. You will be given a manipulation instruction.
3. Apply the manipulation ONLY to the specified sentence(s) (context or target).
4. After manipulation, identify the "hotspots" in the MODIFIED target sentence.
5. Return the result as a single JSON object with the specified schema. Do not add any extra text, explanations, or markdown formatting.

**Manipulation:**
{instruction}

**Input Sentences:**
- Context: "{context}"
- Target: "{target}"

**Hotspots to identify in the MODIFIED target sentence:**
- subject: The subject of the main clause. If the subject is omitted (a "subject drop"), return "Ø".
- verb: The main verb immediately following the subject.
- object: The direct or indirect object pronoun, if one exists. Otherwise, return null.
- spillover1: The first word immediately following the main verb. Return null if not present.
- spillover2: The second word immediately following the main verb. Return null if not present.

Return only a JSON object with this structure:
{{
    "manipulated_context": "the manipulated context sentence",
    "manipulated_target": "the manipulated target sentence", 
    "hotspots": {{
        "subject": "subject or Ø",
        "verb": "main verb",
        "object": "object or null",
        "spillover1": "first spillover or null",
        "spillover2": "second spillover or null"
    }}
}}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(API_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Try to parse as JSON, handling markdown code blocks
        try:
            # Remove markdown code block formatting if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.endswith('```'):
                content = content[:-3]  # Remove ```
            content = content.strip()
            
            parsed_json = json.loads(content)
            return {
                'manipulated_context': parsed_json['manipulated_context'],
                'manipulated_target': parsed_json['manipulated_target'],
                'hotspots': parsed_json['hotspots']
            }
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response: {content}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def transform_stimuli():
    """Transform all non-default stimuli using DeepSeek API"""
    
    # Read the master file
    df = pd.read_csv('long_format/master_stimuli_with_forms.csv', index_col=0)
    
    # Get rows that need transformation (non-default forms)
    transform_rows = df[df['form'] != 'default'].copy()
    
    print(f"Found {len(transform_rows)} rows to transform")
    print(f"Forms to transform: {transform_rows['form'].unique()}")
    
    # Initialize progress bar
    progress_bar = tqdm(total=len(transform_rows), desc="Transforming stimuli", unit="item")
    
    successful_transforms = 0
    failed_transforms = 0
    checkpoint_interval = 50  # Save progress every 50 transforms
    
    for i, (idx, row) in enumerate(transform_rows.iterrows()):
        form = row['form']
        context = row['c_english'].strip()
        target = row['target'].strip()
        
        # Update progress bar description
        progress_bar.set_description(f"Transforming {form} (item {row['within_item_id']}, id {row['item_id']})")
        
        # Call API
        result = call_deepseek_api(context, target, form)
        
        if result:
            # Update the dataframe
            df.loc[idx, 'c_english'] = result['manipulated_context']
            df.loc[idx, 'target'] = result['manipulated_target']
            
            # Update hotspot (assuming it's the main hotspot word)
            if 'verb' in result['hotspots'] and result['hotspots']['verb']:
                df.loc[idx, 'hotspot_english'] = result['hotspots']['verb']
            
            successful_transforms += 1
        else:
            failed_transforms += 1
            print(f"Failed to transform row {idx}: {form}")
        
        progress_bar.update(1)
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_file = f'long_format/checkpoint_{i+1}.csv'
            df.to_csv(checkpoint_file, index_label='index')
            print(f"\nCheckpoint saved: {checkpoint_file}")
        
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    progress_bar.close()
    
    print(f"\nTransformation complete!")
    print(f"Successful: {successful_transforms}")
    print(f"Failed: {failed_transforms}")
    
    # Save the transformed data
    output_file = 'long_format/master_stimuli_transformed.csv'
    df.to_csv(output_file, index_label='index')
    print(f"Saved transformed data to: {output_file}")
    
    return df

if __name__ == "__main__":
    print("Starting stimulus transformation with DeepSeek API...")
    
    # Check if the input file exists
    if not os.path.exists('long_format/master_stimuli_with_forms.csv'):
        print("Error: master_stimuli_with_forms.csv not found!")
        exit(1)
    
    # Run the transformation
    try:
        transformed_df = transform_stimuli()
        print("All done!")
    except KeyboardInterrupt:
        print("\nTransformation interrupted by user")
    except Exception as e:
        print(f"Error during transformation: {e}")