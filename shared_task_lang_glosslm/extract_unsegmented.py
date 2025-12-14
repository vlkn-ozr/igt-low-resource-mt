#!/usr/bin/env python3
"""Extract only unsegmented data (is_segmented == 'no') for target languages."""

import pandas as pd
import os

target_glottocodes = {
    'uspa1245': 'Uspanteko (usp)'
}

dataset_path = 'glosslm-corpus-split/data'

train_df = pd.read_parquet(os.path.join(dataset_path, 'train_ID-00000-of-00001.parquet'))
eval_df = pd.read_parquet(os.path.join(dataset_path, 'eval_ID-00000-of-00001.parquet'))
test_df = pd.read_parquet(os.path.join(dataset_path, 'test_ID-00000-of-00001.parquet'))

print('Checking segmentation status in data:')
print('-' * 60)
for df_name, df in [('train', train_df), ('eval', eval_df), ('test', test_df)]:
    segmented_values = df['is_segmented'].unique()
    print(f"{df_name} segmentation values: {segmented_values}")
    
    yes_count = len(df[df['is_segmented'] == 'yes'])
    no_count = len(df[df['is_segmented'] == 'no'])
    empty_count = len(df[df['is_segmented'] == ''])
    
    print(f"{df_name} - yes: {yes_count}, no: {no_count}, empty: {empty_count}, total: {len(df)}")
    print('-' * 40)

print('\nUnsegmented sample counts by language (only "no"):')
print('-' * 50)
print(f"{'Language':<20} {'Train':<10} {'Eval':<10} {'Test':<10}")
print('-' * 50)

for glottocode, lang_name in target_glottocodes.items():
    train_unseg = train_df[(train_df['glottocode'] == glottocode) & (train_df['is_segmented'] == 'no')]
    eval_unseg = eval_df[(eval_df['glottocode'] == glottocode) & (eval_df['is_segmented'] == 'no')]
    test_unseg = test_df[(test_df['glottocode'] == glottocode) & (test_df['is_segmented'] == 'no')]
    
    train_count = len(train_unseg)
    eval_count = len(eval_unseg)
    test_count = len(test_unseg)
    
    print(f"{lang_name:<20} {train_count:<10} {eval_count:<10} {test_count:<10}")

print('\nExamples of segmented and unsegmented samples:')
print('=' * 80)

for glottocode, lang_name in target_glottocodes.items():
    lang_data = train_df[train_df['glottocode'] == glottocode]
    
    segmented_sample = lang_data[lang_data['is_segmented'] == 'yes'].iloc[0] if not lang_data[lang_data['is_segmented'] == 'yes'].empty else None
    unsegmented_sample = lang_data[lang_data['is_segmented'] == 'no'].iloc[0] if not lang_data[lang_data['is_segmented'] == 'no'].empty else None
    
    print(f"Language: {lang_name}")
    
    if segmented_sample is not None:
        print("  SEGMENTED SAMPLE:")
        print(f"  Transcription: {segmented_sample['transcription']}")
        print(f"  Glosses: {segmented_sample['glosses']}")
        seg_trans_words = len(str(segmented_sample['transcription']).split())
        seg_gloss_words = len(str(segmented_sample['glosses']).split())
        print(f"  Word count - Transcription: {seg_trans_words}, Glosses: {seg_gloss_words}")
    else:
        print("  No segmented samples found")
    
    if unsegmented_sample is not None:
        print("\n  UNSEGMENTED SAMPLE:")
        print(f"  Transcription: {unsegmented_sample['transcription']}")
        print(f"  Glosses: {unsegmented_sample['glosses']}")
        unseg_trans_words = len(str(unsegmented_sample['transcription']).split())
        unseg_gloss_words = len(str(unsegmented_sample['glosses']).split())
        print(f"  Word count - Transcription: {unseg_trans_words}, Glosses: {unseg_gloss_words}")
    else:
        print("  No unsegmented samples found")
    
    print('-' * 80)

print('\nExtracting unsegmented data to files (only "no"):')
os.makedirs('unsegmented_data_strict', exist_ok=True)

for glottocode, lang_name in target_glottocodes.items():
    lang_code = lang_name.split('(')[1].split(')')[0]
    
    lang_train = train_df[(train_df['glottocode'] == glottocode) & (train_df['is_segmented'] == 'no')]
    lang_eval = eval_df[(eval_df['glottocode'] == glottocode) & (eval_df['is_segmented'] == 'no')]
    lang_test = test_df[(test_df['glottocode'] == glottocode) & (test_df['is_segmented'] == 'no')]
    
    out_path = f"unsegmented_data_strict/{lang_code}"
    os.makedirs(out_path, exist_ok=True)
    
    if not lang_train.empty:
        lang_train.to_csv(f"{out_path}/train.csv", index=False)
        print(f"  Saved {len(lang_train)} unsegmented training samples for {lang_name}")
    if not lang_eval.empty:
        lang_eval.to_csv(f"{out_path}/eval.csv", index=False)
        print(f"  Saved {len(lang_eval)} unsegmented eval samples for {lang_name}")
    if not lang_test.empty:
        lang_test.to_csv(f"{out_path}/test.csv", index=False)
        print(f"  Saved {len(lang_test)} unsegmented test samples for {lang_name}")
    
    total = len(lang_train) + len(lang_eval) + len(lang_test)
    if total > 0:
        print(f"  Total: {total} unsegmented samples for {lang_name}")
    else:
        print(f"  No unsegmented samples found for {lang_name}")

print('\nDone!') 