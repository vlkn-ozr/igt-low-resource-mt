#!/usr/bin/env python3
"""Extract data for target languages from GlossLM corpus split dataset."""

import pandas as pd
import os

target_glottocodes = {
    'uspa1245': 'Uspanteko (usp)'
}

dataset_path = 'glosslm-corpus-split/data'

train_df = pd.read_parquet(os.path.join(dataset_path, 'train_ID-00000-of-00001.parquet'))
eval_df = pd.read_parquet(os.path.join(dataset_path, 'eval_ID-00000-of-00001.parquet'))
test_df = pd.read_parquet(os.path.join(dataset_path, 'test_ID-00000-of-00001.parquet'))

print('Out-of-domain languages data counts:')
print('-' * 50)
print(f"{'Language':<20} {'Train':<10} {'Eval':<10} {'Test':<10}")
print('-' * 50)

for glottocode, lang_name in target_glottocodes.items():
    train_count = len(train_df[train_df['glottocode'] == glottocode])
    eval_count = len(eval_df[eval_df['glottocode'] == glottocode]) 
    test_count = len(test_df[test_df['glottocode'] == glottocode])
    
    print(f"{lang_name:<20} {train_count:<10} {eval_count:<10} {test_count:<10}")

print('\nSample data for each language:')
print('=' * 80)

for glottocode, lang_name in target_glottocodes.items():
    if len(train_df[train_df['glottocode'] == glottocode]) > 0:
        sample = train_df[train_df['glottocode'] == glottocode].iloc[0]
    elif len(eval_df[eval_df['glottocode'] == glottocode]) > 0:
        sample = eval_df[eval_df['glottocode'] == glottocode].iloc[0]
    elif len(test_df[test_df['glottocode'] == glottocode]) > 0:
        sample = test_df[test_df['glottocode'] == glottocode].iloc[0]
    else:
        print(f"No data found for {lang_name}")
        continue
    
    print(f"Sample for {lang_name}:")
    print(f"  Transcription: {sample['transcription']}")
    print(f"  Glosses: {sample['glosses']}")
    print(f"  Translation: {sample['translation']}")
    print(f"  Metalanguage: {sample['metalang']}")
    print(f"  Segmented: {sample['is_segmented']}")
    print('-' * 80)

print('\nExtracting data to files...')
os.makedirs('extracted_data', exist_ok=True)

for glottocode, lang_name in target_glottocodes.items():
    lang_code = lang_name.split('(')[1].split(')')[0]
    
    lang_train = train_df[train_df['glottocode'] == glottocode]
    lang_eval = eval_df[eval_df['glottocode'] == glottocode]
    lang_test = test_df[test_df['glottocode'] == glottocode]
    
    out_path = f"extracted_data/{lang_code}"
    os.makedirs(out_path, exist_ok=True)
    
    if not lang_train.empty:
        lang_train.to_csv(f"{out_path}/train.csv", index=False)
    if not lang_eval.empty:
        lang_eval.to_csv(f"{out_path}/eval.csv", index=False)
    if not lang_test.empty:
        lang_test.to_csv(f"{out_path}/test.csv", index=False)
    
    print(f"  Saved data for {lang_name} to {out_path}/")

print('\nDone!') 