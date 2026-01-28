import pandas as pd
import spacy
import os
import sys
from pathlib import Path
import random
from typing import List, Tuple, Dict
from collections import defaultdict

SPACY_MODELS = {
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm', 
    'it': 'it_core_news_sm',
    'es': 'es_core_news_sm'
}

class ImprovedCodeSwitcher:    
    def __init__(self):
        self.nlp = {}
        print("Init spaCy models")
        for lang, model in SPACY_MODELS.items():
            self.nlp[lang] = spacy.load(model)
    
    def strategy_word_by_word_random(self, sent1: str, lang1: str,
                                    sent2: str, lang2: str,
                                    target_ratio: float = 0.5,
                                    seed: int = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        doc1 = self.nlp[lang1](sent1)
        doc2 = self.nlp[lang2](sent2)
        
        # Use the longer sentence as base
        if len(doc1) >= len(doc2):
            base_doc = doc1
            alt_doc = doc2
            base_lang = lang1
            alt_lang = lang2
        else:
            base_doc = doc2
            alt_doc = doc1
            base_lang = lang2
            alt_lang = lang1
            target_ratio = 1 - target_ratio  # Flip ratio
        
        result = []
        lang_counts = {base_lang: 0, alt_lang: 0}
        
        for i, base_token in enumerate(base_doc):
            # Decide which language to use
            if i < len(alt_doc):
                # Use target ratio to decide
                if random.random() < target_ratio:
                    result.append(alt_doc[i].text)
                    lang_counts[alt_lang] += 1
                else:
                    result.append(base_token.text)
                    lang_counts[base_lang] += 1
            else:
                result.append(base_token.text)
                lang_counts[base_lang] += 1
        
        # Calculate actual ratio
        total = sum(lang_counts.values())
        actual_ratio = lang_counts[alt_lang] / total if total > 0 else 0
        
        return ' '.join(result), lang_counts, actual_ratio
    
    def strategy_pos_aware_mixing(self, sent1: str, lang1: str,
                                  sent2: str, lang2: str,
                                  seed: int = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        doc1 = self.nlp[lang1](sent1)
        doc2 = self.nlp[lang2](sent2)
        
        # Collect tokens by POS type
        pos_groups1 = defaultdict(list)
        pos_groups2 = defaultdict(list)
        
        for token in doc1:
            pos_groups1[token.pos_].append(token.text)
        
        for token in doc2:
            pos_groups2[token.pos_].append(token.text)
        
        # All POS types present in either sentence
        all_pos = set(pos_groups1.keys()) | set(pos_groups2.keys())
        
        # For each POS type, randomly choose which language to use
        result_groups = {}
        lang_counts = {lang1: 0, lang2: 0}
        
        for pos in all_pos:
            tokens1 = pos_groups1.get(pos, [])
            tokens2 = pos_groups2.get(pos, [])
            
            if tokens1 and tokens2:
                # Both have this POS - randomly pick language
                if random.random() < 0.5:
                    result_groups[pos] = (tokens1, lang1)
                    lang_counts[lang1] += len(tokens1)
                else:
                    result_groups[pos] = (tokens2, lang2)
                    lang_counts[lang2] += len(tokens2)
            elif tokens1:
                result_groups[pos] = (tokens1, lang1)
                lang_counts[lang1] += len(tokens1)
            else:
                result_groups[pos] = (tokens2, lang2)
                lang_counts[lang2] += len(tokens2)
        
        # Reconstruct sentence maintaining original order (using doc1 as template)
        result = []
        for token in doc1:
            if token.pos_ in result_groups and result_groups[token.pos_][0]:
                # Pop from the group
                chosen_tokens, _ = result_groups[token.pos_]
                if chosen_tokens:
                    result.append(chosen_tokens.pop(0))
                else:
                    result.append(token.text)
            else:
                result.append(token.text)
        
        total = sum(lang_counts.values())
        ratio = lang_counts[lang2] / total if total > 0 else 0
        
        return ' '.join(result), lang_counts, ratio
    
    def strategy_chunk_based(self, sent1: str, lang1: str,
                            sent2: str, lang2: str,
                            chunk_size: int = 2,
                            seed: int = None) -> str:
        if seed is not None:
            random.seed(seed)
        
        doc1 = self.nlp[lang1](sent1)
        doc2 = self.nlp[lang2](sent2)
        
        tokens1 = [t.text for t in doc1]
        tokens2 = [t.text for t in doc2]
        
        result = []
        lang_counts = {lang1: 0, lang2: 0}
        
        i1, i2 = 0, 0
        use_lang1 = random.choice([True, False])  # Random start
        
        while i1 < len(tokens1) or i2 < len(tokens2):
            if use_lang1 and i1 < len(tokens1):
                # Take chunk from lang1
                for _ in range(chunk_size):
                    if i1 < len(tokens1):
                        result.append(tokens1[i1])
                        lang_counts[lang1] += 1
                        i1 += 1
            elif not use_lang1 and i2 < len(tokens2):
                # Take chunk from lang2
                for _ in range(chunk_size):
                    if i2 < len(tokens2):
                        result.append(tokens2[i2])
                        lang_counts[lang2] += 1
                        i2 += 1
            elif i1 < len(tokens1):
                # Fallback to lang1
                result.append(tokens1[i1])
                lang_counts[lang1] += 1
                i1 += 1
            elif i2 < len(tokens2):
                # Fallback to lang2
                result.append(tokens2[i2])
                lang_counts[lang2] += 1
                i2 += 1
            
            use_lang1 = not use_lang1  # Alternate
        
        total = sum(lang_counts.values())
        ratio = lang_counts[lang2] / total if total > 0 else 0
        
        return ' '.join(result), lang_counts, ratio
    
    def generate_code_switch(self, sent1: str, lang1: str,
                           sent2: str, lang2: str,
                           strategy: str = 'random',
                           seed: int = None) -> Tuple[str, Dict, float]:
        if strategy == 'random':
            return self.strategy_word_by_word_random(sent1, lang1, sent2, lang2, 0.5, seed)
        elif strategy == 'pos_aware':
            return self.strategy_pos_aware_mixing(sent1, lang1, sent2, lang2, seed)
        elif strategy == 'chunk':
            return self.strategy_chunk_based(sent1, lang1, sent2, lang2, 2, seed)
        else:
            # Default to random
            return self.strategy_word_by_word_random(sent1, lang1, sent2, lang2, 0.5, seed)


def process_csv_files(input_dir: str = "translations", 
                     output_dir: str = "code_switched_output",
                     strategy: str = 'random',
                     use_seed: bool = True):
    
    switcher = ImprovedCodeSwitcher()
    Path(output_dir).mkdir(exist_ok=True)
    
    lang_pairs = [
        ('fr', 'de'),
        ('fr', 'it'),
        ('fr', 'es'),
        ('de', 'it'),
        ('de', 'es'),
        ('it', 'es')
    ]
    
    print("Loading CSV files from:", input_dir)
    print("Strategy:", strategy)
    
    dataframes = {}
    for lang in ['de', 'fr', 'it', 'es']:
        csv_path = os.path.join(input_dir, f"jbb_behaviors_{lang}_12b.csv")
        if os.path.exists(csv_path):
            dataframes[lang] = pd.read_csv(csv_path)
            print(f"✓ Loaded {csv_path}: {len(dataframes[lang])} rows")
        else:
            print(f"✗ NOT FOUND: {csv_path}")
    
    if len(dataframes) < 2:
        print("\nERROR: Need at least 2 CSV files")
        sys.exit(1)
    
    total_processed = 0
    mixing_stats = []
    
    for lang1, lang2 in lang_pairs:
        if lang1 not in dataframes or lang2 not in dataframes:
            print(f"\n✗ Skipping {lang1}-{lang2}: Missing data files")
            continue
        print(f"\nProcessing language pair: {lang1}-{lang2}")        
        df1 = dataframes[lang1]
        df2 = dataframes[lang2]
        
        # Find Goal columns
        goal_col1 = f"Goal_{lang1}"
        goal_col2 = f"Goal_{lang2}"
        
        if goal_col1 not in df1.columns:
            possible_cols = [c for c in df1.columns if 'goal' in c.lower()]
            if possible_cols:
                goal_col1 = possible_cols[0]
        
        if goal_col2 not in df2.columns:
            possible_cols = [c for c in df2.columns if 'goal' in c.lower()]
            if possible_cols:
                goal_col2 = possible_cols[0]
        
        results = []
        num_prompts = min(len(df1), len(df2))
        
        print(f"  Processing {num_prompts} prompts...")
        
        for idx in range(num_prompts):
            sent1 = str(df1.iloc[idx][goal_col1])
            sent2 = str(df2.iloc[idx][goal_col2])
            
            if pd.isna(sent1) or pd.isna(sent2) or sent1 == 'nan' or sent2 == 'nan':
                continue
            
            # Generate with mixing stats
            seed = idx if use_seed else None
            cs_sent, lang_counts, mix_ratio = switcher.generate_code_switch(
                sent1, lang1, sent2, lang2, strategy, seed
            )
            
            # Store mixing ratio for statistics
            mixing_stats.append(mix_ratio * 100)
            
            results.append({
                'prompt_id': idx,
                f'original_{lang1}': sent1,
                f'original_{lang2}': sent2,
                'code_switched': cs_sent,
                f'{lang1}_count': lang_counts[lang1],
                f'{lang2}_count': lang_counts[lang2],
                f'{lang2}_percentage': round(mix_ratio * 100, 1),
                'strategy': strategy,
                'lang_pair': f'{lang1}-{lang2}'
            })
            
            if (idx + 1) % 25 == 0:
                print(f"    Progress: {idx + 1}/{num_prompts}")
        
        if results:
            output_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f"code_switched_{lang1}_{lang2}.csv")
            output_df.to_csv(output_path, index=False)
            
            # Calculate statistics for this pair
            pair_ratios = [r[f'{lang2}_percentage'] for r in results]
            avg_mix = sum(pair_ratios) / len(pair_ratios)
            min_mix = min(pair_ratios)
            max_mix = max(pair_ratios)
            
            print(f"  ✓ Saved {len(results)} prompts")
            print(f"    Mixing stats - Avg: {avg_mix:.1f}%, Min: {min_mix:.1f}%, Max: {max_mix:.1f}%")
            print(f"    Output: {output_path}")
            total_processed += len(results)
    
    # Overall statistics
    if mixing_stats:
        avg_overall = sum(mixing_stats) / len(mixing_stats)
        print("PROCESSING COMPLETE")
        print(f"Total code-switched prompts: {total_processed}")
        print(f"Overall average mixing: {avg_overall:.1f}%")
        print(f"Range: {min(mixing_stats):.1f}% - {max(mixing_stats):.1f}%")


def main():
    
    INPUT_DIR = "translations"
    OUTPUT_DIR = "code_switched_output"
    STRATEGY = "chunk"  # 'random', 'pos_aware', 'chunk'

    print(f"Input:    {INPUT_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Strategy: {STRATEGY}")
    
    process_csv_files(INPUT_DIR, OUTPUT_DIR, STRATEGY, use_seed=True)


if __name__ == "__main__":
    main()