import pandas as pd
import spacy
import os
import sys
from pathlib import Path
import random
from typing import List, Tuple

# Language models
SPACY_MODELS = {
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm', 
    'it': 'it_core_news_sm',
    'es': 'es_core_news_sm'
}

class MatrixLanguageFrameCodeSwitcher:
    
    def __init__(self):
        """Initialize spaCy models"""
        self.nlp = {}
        print("Loading spaCy models...")
        for lang, model in SPACY_MODELS.items():
            self.nlp[lang] = spacy.load(model)
            print(f"  ✓ Loaded {model}")
    
    def generate_code_switch(self, sent1: str, lang1: str, 
                            sent2: str, lang2: str,
                            seed: int = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        doc1 = self.nlp[lang1](sent1)
        doc2 = self.nlp[lang2](sent2)
        
        # Content words that can be switched
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
        
        result = []
        
        for i, token1 in enumerate(doc1):
            if i < len(doc2):
                token2 = doc2[i]
                
                # If both are content words and same POS, 50% chance to switch
                if (token1.pos_ in content_pos and 
                    token1.pos_ == token2.pos_ and 
                    random.random() > 0.5):
                    result.append(token2.text)
                else:
                    # Keep original from matrix language
                    result.append(token1.text)
            else:
                result.append(token1.text)
        
        return ' '.join(result)


def process_csv_files(input_dir: str = "translations", 
                     output_dir: str = "code_switched_output",
                     use_seed: bool = True):
    
    # Initialize code switcher
    switcher = MatrixLanguageFrameCodeSwitcher()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Language pairs 
    lang_pairs = [
        ('fr', 'de'),
        ('fr', 'it'),
        ('fr', 'es'),
        ('de', 'it'),
        ('de', 'es'),
        ('it', 'es')
    ]
    
    # Load CSV files
    print("\n" + "="*70)
    print("Loading CSV files from:", input_dir)
    print("="*70)
    
    dataframes = {}
    for lang in ['de', 'fr', 'it', 'es']:
        csv_path = os.path.join(input_dir, f"jbb_behaviors_{lang}_12b.csv")
        if os.path.exists(csv_path):
            dataframes[lang] = pd.read_csv(csv_path)
            print(f"Loaded {csv_path}: {len(dataframes[lang])} rows")
        else:
            print(f"NOT FOUND: {csv_path}")
    
    if len(dataframes) < 2:
        print("\nERROR: Need at least 2 CSV files to create code-switched pairs")
        print(f"Expected files in '{input_dir}/':")
        for lang in ['de', 'fr', 'it', 'es']:
            print(f"  - jbb_behaviors_{lang}_12b.csv")
        sys.exit(1)
    
    # Process each language pair
    total_processed = 0
    
    for lang1, lang2 in lang_pairs:
        if lang1 not in dataframes or lang2 not in dataframes:
            print(f"\n Skipping {lang1}-{lang2}: Missing data files")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {lang1.upper()}-{lang2.upper()}")
        print(f"{'='*70}")
        
        df1 = dataframes[lang1]
        df2 = dataframes[lang2]
        
        # Find Goal columns
        goal_col1 = f"Goal_{lang1}"
        goal_col2 = f"Goal_{lang2}"
        
        # Check for alternative column names
        if goal_col1 not in df1.columns:
            # Try other common patterns
            possible_cols = [c for c in df1.columns if 'goal' in c.lower()]
            if possible_cols:
                goal_col1 = possible_cols[0]
                print(f"  Using column: {goal_col1}")
            else:
                print(f"  ERROR: No Goal column found in {lang1} file")
                print(f"  Available columns: {df1.columns.tolist()}")
                continue
        
        if goal_col2 not in df2.columns:
            possible_cols = [c for c in df2.columns if 'goal' in c.lower()]
            if possible_cols:
                goal_col2 = possible_cols[0]
                print(f"  Using column: {goal_col2}")
            else:
                print(f"  ERROR: No Goal column found in {lang2} file")
                print(f"  Available columns: {df2.columns.tolist()}")
                continue
        
        # Process prompts
        results = []
        num_prompts = min(len(df1), len(df2))
        
        print(f"  Processing {num_prompts} prompts...")
        
        for idx in range(num_prompts):
            sent1 = str(df1.iloc[idx][goal_col1])
            sent2 = str(df2.iloc[idx][goal_col2])
            
            # Skip invalid entries
            if pd.isna(sent1) or pd.isna(sent2) or sent1 == 'nan' or sent2 == 'nan':
                continue
            
            # Generate code-switched version
            seed = idx if use_seed else None
            cs_sent = switcher.generate_code_switch(sent1, lang1, sent2, lang2, seed)
            
            # Store result
            results.append({
                'prompt_id': idx,
                f'original_{lang1}': sent1,
                f'original_{lang2}': sent2,
                'code_switched': cs_sent,
                'strategy': 'matrix_language_frame',
                'lang_pair': f'{lang1}-{lang2}'
            })
            
            # Progress indicator
            if (idx + 1) % 25 == 0:
                print(f"    Progress: {idx + 1}/{num_prompts}")
        
        # Save results
        if results:
            output_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f"code_switched_{lang1}_{lang2}.csv")
            output_df.to_csv(output_path, index=False)
            print(f"  ✓ Saved {len(results)} code-switched prompts")
            print(f"    Output: {output_path}")
            total_processed += len(results)
        else:
            print(f"  ✗ No valid prompts found")
    
    # Summary
    print(f"\nTotal code-switched prompts generated: {total_processed}")
    print(f"Output directory: {output_dir}/")
    print(f"\nGenerated files:")
    for f in sorted(Path(output_dir).glob("code_switched_*.csv")):
        print(f"  - {f.name}")


def main():    
    # Configuration
    INPUT_DIR = "translations"
    OUTPUT_DIR = "code_switched_output"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
    print("\nCODE-SWITCHING GENERATOR")
    print("Strategy: Matrix Language Frame (Best performing)")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process files
    process_csv_files(INPUT_DIR, OUTPUT_DIR, use_seed=True)


if __name__ == "__main__":
    main()