import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent


def collect_json_files(pattern: str = "translation_results_*.json") -> List[Path]:
    """Return all result JSON files created by eval_glosslm."""
    return sorted(ROOT.glob(pattern))


def write_predictions(json_path: Path):
    """Load a result JSON and dump its generated lines into the language folder."""
    lang_code = json_path.stem.split("_")[-1]  # e.g., translation_results_usp -> usp
    out_dir = ROOT / "test_set" / lang_code
    if not out_dir.exists():
        print(f"[WARN] Language folder {out_dir} does not exist. Skipping.")
        return

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    zero_shot = data.get("zero_shot", [])
    preds = [ex.get("generated", "").strip() for ex in zero_shot]

    if not preds:
        print(f"[WARN] No predictions found in {json_path}. Skipping.")
        return

    out_file = out_dir / "predicted.glosses.txt"
    out_file.write_text("\n".join(preds), encoding="utf-8")
    print(f"Saved {len(preds)} predictions to {out_file.relative_to(ROOT)}")


def main():
    json_files = collect_json_files()
    if not json_files:
        print("No translation_results_<lang>.json files found.")
        return

    for jf in json_files:
        write_predictions(jf)


if __name__ == "__main__":
    main() 