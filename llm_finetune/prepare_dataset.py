import json
import os

def prepare_dataset(gloss_file, translation_file, output_file):
    """Prepare a dataset for fine-tuning Qwen models on gloss-to-translation pairs."""
    with open(gloss_file, 'r', encoding='utf-8') as f:
        glosses = [line.strip() for line in f.readlines()]
    
    with open(translation_file, 'r', encoding='utf-8') as f:
        translations = [line.strip() for line in f.readlines()]
    
    assert len(glosses) == len(translations), "Number of glosses and translations must match"
    
    dataset = []
    for gloss, translation in zip(glosses, translations):
        sample = {
            "messages": [
                {"role": "user", "content": f"Translate the following linguistic gloss to English: {gloss}"},
                {"role": "assistant", "content": translation}
            ]
        }
        dataset.append(sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)
    
    prepare_dataset(
        gloss_file="data/gloss.txt",
        translation_file="data/translation.txt",
        output_file="processed/gloss_translation_dataset.jsonl"
    ) 