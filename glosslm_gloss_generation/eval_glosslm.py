import json
from pathlib import Path
from typing import List, Dict

from tqdm.auto import tqdm
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

LANGUAGE_NAMES = {
    "usp": "Uspanteco",
    "git": "Gitxsan",
    "lez": "Lezgian",
    "ddo": "Tsez",
    "ntu": "Natügu",
}


def build_prompt(transcription: str, translation: str, lang_code: str) -> str:
    """Construct the prompt expected by GlossLM (from its model card)."""
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
    metalang = "Spanish" if lang_code == "usp" else "English"

    return (
        f"""Provide the glosses for the following transcription in {lang_name}.

Transcription in {lang_name}: {transcription}
Transcription segmented: False
Translation in {metalang}: {translation}\n
Glosses: """
    )


def evaluate_language_folder(folder: Path, model, tokenizer):
    """Run zero-shot gloss generation for a single language folder and write its own results file."""
    gloss_path = folder / "test.glosses.txt"
    transcription_path = folder / "test.transcription.txt"
    translation_path = folder / "test.translation.txt"

    if not all(p.exists() for p in (gloss_path, transcription_path, translation_path)):
        print(f"[WARN] Missing required files in {folder}. Skipping.")
        return

    output_file = f"translation_results_{folder.name}.json"
    results: List[Dict[str, str]] = []

    gloss_lines = gloss_path.read_text(encoding="utf-8").splitlines()
    transcription_lines = transcription_path.read_text(encoding="utf-8").splitlines()
    translation_lines = translation_path.read_text(encoding="utf-8").splitlines()

    n = min(len(gloss_lines), len(transcription_lines), len(translation_lines))
    if n == 0:
        print(f"[WARN] No aligned examples in {folder}.")
        return

    for gloss, transcrip, transl in tqdm(
        list(zip(gloss_lines[:n], transcription_lines[:n], translation_lines[:n])),
        total=n,
        desc=folder.name,
    ):
        prompt = build_prompt(transcrip, transl, folder.name.lower())
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=1024)
        generated = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        results.append({
            "input": transcrip,
            "expected": gloss,
            "generated": generated.strip(),
        })

        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump({"zero_shot": results}, fh, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file}")


def main(test_root: str = "test_set"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}…")
    model = T5ForConditionalGeneration.from_pretrained("lecslab/glosslm").to(device)
    tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-base", use_fast=False)

    for folder in sorted(Path(test_root).iterdir()):
        if folder.is_dir():
            evaluate_language_folder(folder, model, tokenizer)

    print("All languages processed.")


if __name__ == "__main__":
    main() 