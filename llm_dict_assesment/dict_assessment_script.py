#!/usr/bin/env python3
"""
Turkish-English Dictionary Alignment Assessment Script
Uses Qwen2.5-7b-instruct model to assess whether Turkish-English dictionary pairs are correctly aligned.
When incorrect, the model suggests the correct English translation.
"""

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from datetime import datetime

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Please install required packages: pip install transformers torch accelerate")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dict_assessment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DictAssessment:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the dictionary assessment system."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.load_model()
        
        self.results = []
        self.correct_pairs = []
        self.incorrect_pairs = []
        self.corrected_pairs = []
        
    def load_model(self):
        """Load the Qwen model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
                
            self.model.eval()
                
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_assessment_prompt(self, source: str, target: str) -> str:
        """Create a concise prompt for assessing Turkish-English dictionary alignment."""
        prompt = f"""Evaluate this Turkish-English lemma pair for correctness:

Turkish: "{source}"
English: "{target}"

RULES:
1. Numbers must stay numbers (1000 -> 1000, NOT 1000 -> one_thousand)
2. Symbols stay identical (& -> &, / -> /)
3. Abbreviations stay abbreviations (akp -> akp, nato -> nato)
4. Proper names stay identical (ahmet -> ahmet, ankara -> ankara)
5. Unknown abbreviations/codes stay identical when uncertain
6. Only correct clear translation errors

EXAMPLES:
- "123 -> 456" = INCORRECT, correction: 123
- "ev -> house" = CORRECT
- "köpek -> car" = INCORRECT, correction: dog
- "akp -> akp" = CORRECT (political party)
- "& -> &" = CORRECT (symbol)
- "ab -> eu" = CORRECT (European Union abbreviation)

FORMAT:
If correct: <VERDICT>CORRECT</VERDICT>
If incorrect: <VERDICT>INCORRECT</VERDICT><CORRECTION>actual_correct_translation</CORRECTION>

Respond only with the format above, no explanations.
"""
        return prompt
    
    def parse_dict_file(self, file_path: str) -> List[Tuple[str, str]]:
        """Parse the dictionary file and extract source-target pairs."""
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    if ' -> ' in line:
                        source, target = line.split(' -> ', 1)
                        pairs.append((source.strip(), target.strip()))
                    else:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
            
            logger.info(f"Loaded {len(pairs)} dictionary pairs")
            return pairs
        except Exception as e:
            logger.error(f"Error reading dictionary file: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from the model."""
        try:
            messages = [
                {"role": "system", "content": "You are a Turkish-English dictionary expert. Evaluate pairs precisely. When uncertain about abbreviations or proper nouns, keep them identical. Only correct obvious translation errors."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer(
                [text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs.get('attention_mask', None),
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
                    use_cache=True
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def extract_verdict_and_correction(self, response: str) -> Tuple[str, str]:
        """Extract the verdict and correction (if any) from the model response."""
        response = response.strip()
        
        verdict_match = re.search(r'<VERDICT>(CORRECT|INCORRECT)</VERDICT>', response, re.IGNORECASE)
        verdict = verdict_match.group(1).upper() if verdict_match else "UNCLEAR"
        
        correction_match = re.search(r'<CORRECTION>(.*?)</CORRECTION>', response, re.IGNORECASE | re.DOTALL)
        correction = correction_match.group(1).strip() if correction_match else ""
        
        if verdict == "UNCLEAR":
            response_lower = response.lower()
            if response_lower.startswith('correct'):
                verdict = "CORRECT"
            elif response_lower.startswith('incorrect'):
                verdict = "INCORRECT"
            elif 'correct' in response_lower and 'incorrect' not in response_lower:
                verdict = "CORRECT"
            elif 'incorrect' in response_lower and 'correct' not in response_lower:
                verdict = "INCORRECT"
            else:
                logger.warning(f"Could not extract verdict from response: {response[:100]}...")
        
        if verdict == "INCORRECT" and not correction:
            remaining_text = re.sub(r'<VERDICT>INCORRECT</VERDICT>', '', response, flags=re.IGNORECASE).strip()
            if remaining_text and not remaining_text.startswith('<'):
                correction = remaining_text
                logger.info(f"Extracted correction from remaining text: '{correction}'")
        
        if correction and verdict == "INCORRECT":
            if "correct_english_lemma" in correction.lower() or "actual_correct_translation" in correction.lower():
                logger.warning(f"Suspicious correction detected (placeholder text): '{correction}' - setting to empty")
                correction = ""
            elif len(correction) > 50:
                logger.warning(f"Suspicious correction detected (too long): '{correction[:50]}...' - setting to empty")
                correction = ""
            elif any(word in correction.lower() for word in ["this is", "the correct", "should be", "translation", "meaning"]):
                logger.warning(f"Suspicious correction detected (explanation): '{correction}' - setting to empty")
                correction = ""
        
        return verdict, correction
    
    def validate_and_fix_assessment(self, source: str, target: str, verdict: str, correction: str) -> Tuple[str, str]:
        """Validate the assessment and apply fixes for common issues."""
        original_verdict = verdict
        original_correction = correction
        
        if source == target:
            if verdict != "CORRECT":
                logger.info(f"Fixed identical pair: '{source}' -> '{target}' from {verdict} to CORRECT")
                verdict = "CORRECT"
                correction = ""
        
        if source.isdigit() and correction and not correction.isdigit():
            logger.warning(f"Number correction invalid: '{source}' -> '{correction}' (not a number)")
            correction = source
        
        if len(source) == 1 and source in "!@#$%^&*()_+-=[]{}|;':\",./<>?":
            if target != source:
                logger.info(f"Symbol pair corrected to identical: '{source}' -> '{target}' to '{source}' -> '{source}'")
                verdict = "INCORRECT"
                correction = source
        
        turkish_abbrevs = {"akp", "chp", "mhp", "hdp", "nato", "ab", "eu", "afp", "reuters"}
        if source.lower() in turkish_abbrevs and target.lower() != source.lower():
            if len(target) > 10 or not target.isalpha():
                logger.info(f"Turkish abbreviation corrected: '{source}' -> '{target}' to '{source}' -> '{source}'")
                verdict = "INCORRECT"
                correction = source.lower()
        
        if source[0].isupper() and source.isalpha() and len(source) > 2:
            if target.lower() != source.lower() and not any(char in target.lower() for char in source.lower()[:3]):
                logger.info(f"Proper name likely corrected to identical: '{source}' -> '{target}' to '{source}' -> '{source.lower()}'")
                verdict = "INCORRECT"
                correction = source.lower()
        
        if verdict != original_verdict or correction != original_correction:
            logger.info(f"Assessment validation changed: {original_verdict}:{original_correction} -> {verdict}:{correction}")
        
        return verdict, correction
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free up memory."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def handle_generation_error(self, source: str, target: str, error: Exception) -> Dict:
        """Handle generation errors gracefully."""
        logger.warning(f"Generation failed for '{source}' -> '{target}': {error}")
        
        self.clear_gpu_cache()
        
        if source == target:
            verdict = "CORRECT"
            correction = ""
        else:
            verdict = "UNCLEAR"
            correction = ""
            
        return {
            "source": source,
            "target": target,
            "prompt": "ERROR: Generation failed",
            "model_response": f"ERROR: {str(error)}",
            "verdict": verdict,
            "correction": correction,
            "timestamp": datetime.now().isoformat()
        }
    
    def assess_pair(self, source: str, target: str) -> Dict:
        """Assess a single dictionary pair."""
        try:
            prompt = self.create_assessment_prompt(source, target)
            response = self.generate_response(prompt)
            
            if not response:
                return self.handle_generation_error(source, target, Exception("Empty response"))
                
            verdict, correction = self.extract_verdict_and_correction(response)
            
            verdict, correction = self.validate_and_fix_assessment(source, target, verdict, correction)
            
            result = {
                "source": source,
                "target": target,
                "prompt": prompt,
                "model_response": response,
                "verdict": verdict,
                "correction": correction,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return self.handle_generation_error(source, target, e)
    
    def load_progress(self):
        """Load previous progress from partial results file."""
        results_dir = Path("results")
        progress_file = results_dir / "assessment_results_partial.json"
        
        if not progress_file.exists():
            logger.info("No previous progress file found, starting fresh")
            return 0
        
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            
            self.correct_pairs = []
            self.incorrect_pairs = []
            self.corrected_pairs = []
            
            for result in self.results:
                source = result["source"]
                target = result["target"]
                verdict = result["verdict"]
                correction = result.get("correction", "")
                
                if verdict == "CORRECT":
                    self.correct_pairs.append((source, target))
                elif verdict == "INCORRECT":
                    self.incorrect_pairs.append((source, target))
                    if correction:
                        self.corrected_pairs.append((source, correction))
            
            processed_pairs = len(self.results)
            logger.info(f"Loaded previous progress: {processed_pairs} pairs processed")
            logger.info(f"  - Correct: {len(self.correct_pairs)}")
            logger.info(f"  - Incorrect: {len(self.incorrect_pairs)}")
            logger.info(f"  - With corrections: {len(self.corrected_pairs)}")
            
            return processed_pairs
            
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
            logger.info("Starting fresh due to progress loading error")
            return 0

    def process_dictionary(self, dict_file: str, batch_size: int = 10, max_pairs: int = None, resume_from_line: int = None):
        """Process the entire dictionary file."""
        pairs = self.parse_dict_file(dict_file)
        
        start_index = 0
        if resume_from_line is not None:
            start_index = resume_from_line - 1
            if start_index < 0:
                start_index = 0
            logger.info(f"Resuming from line {resume_from_line} (index {start_index})")
        else:
            processed_count = self.load_progress()
            start_index = processed_count
            if processed_count > 0:
                logger.info(f"Automatically resuming from line {processed_count + 1}")
        
        if start_index >= len(pairs):
            logger.warning(f"Start index {start_index} is beyond dictionary length {len(pairs)}")
            logger.info("Dictionary processing already completed")
            return
        
        if max_pairs:
            end_index = min(start_index + max_pairs, len(pairs))
            pairs = pairs[start_index:end_index]
            total_pairs = len(pairs)
            logger.info(f"Processing {total_pairs} pairs (lines {start_index + 1} to {end_index})")
        else:
            pairs = pairs[start_index:]
            total_pairs = len(pairs)
            logger.info(f"Processing {total_pairs} pairs (lines {start_index + 1} to {start_index + total_pairs})")
        
        logger.info(f"Starting assessment of {total_pairs} pairs")
        
        for i, (source, target) in enumerate(pairs, 1):
            actual_line_number = start_index + i
            
            try:
                logger.info(f"Processing pair {actual_line_number}/{start_index + total_pairs}: '{source}' -> '{target}'")
                
                result = self.assess_pair(source, target)
                result["line_number"] = actual_line_number
                self.results.append(result)
                
                if result["verdict"] == "CORRECT":
                    self.correct_pairs.append((source, target))
                elif result["verdict"] == "INCORRECT":
                    self.incorrect_pairs.append((source, target))
                    if result["correction"]:
                        self.corrected_pairs.append((source, result["correction"]))
                        logger.info(f"  Correction suggested: '{source}' -> '{result['correction']}'")
                
                if i % batch_size == 0:
                    self.save_progress(actual_line_number)
                    self.clear_gpu_cache()
                    logger.info(f"Progress saved at line {actual_line_number}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing pair {actual_line_number}: {source} -> {target}: {e}")
                continue
        
        final_line_number = start_index + total_pairs
        self.save_results(final_line_number)
        logger.info("Dictionary assessment completed")

    def save_progress(self, current_line_number: int = None):
        """Save current progress to files."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        progress_file = results_dir / "assessment_results_partial.json"
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        progress_info_file = results_dir / "progress_info.json"
        progress_info = {
            "last_processed_line": current_line_number,
            "total_processed": len(self.results),
            "last_updated": datetime.now().isoformat(),
            "model_name": self.model_name
        }
        with open(progress_info_file, "w", encoding="utf-8") as f:
            json.dump(progress_info, f, indent=2, ensure_ascii=False)
        
        correct_file = results_dir / "dict_correct_pairs_partial.txt"
        with open(correct_file, "w", encoding="utf-8") as f:
            for source, target in self.correct_pairs:
                f.write(f"{source} -> {target}\n")
        
        incorrect_file = results_dir / "dict_incorrect_pairs_partial.txt"
        with open(incorrect_file, "w", encoding="utf-8") as f:
            for source, target in self.incorrect_pairs:
                f.write(f"{source} -> {target}\n")
        
        corrected_file = results_dir / "dict_corrected_pairs_partial.txt"
        with open(corrected_file, "w", encoding="utf-8") as f:
            for source, corrected_target in self.corrected_pairs:
                f.write(f"{source} -> {corrected_target}\n")
        
        clean_file = results_dir / "dict_clean_combined_partial.txt"
        with open(clean_file, "w", encoding="utf-8") as f:
            for source, target in self.correct_pairs:
                f.write(f"{source} -> {target}\n")
            for source, corrected_target in self.corrected_pairs:
                f.write(f"{source} -> {corrected_target}\n")
        
        summary_file = results_dir / "assessment_summary_partial.json"
        summary = {
            "processed_pairs": len(self.results),
            "last_processed_line": current_line_number,
            "correct_pairs": len(self.correct_pairs),
            "incorrect_pairs": len(self.incorrect_pairs),
            "corrected_pairs": len(self.corrected_pairs),
            "unclear_pairs": len(self.results) - len(self.correct_pairs) - len(self.incorrect_pairs),
            "accuracy_rate": len(self.correct_pairs) / len(self.results) if self.results else 0,
            "correction_rate": len(self.corrected_pairs) / len(self.incorrect_pairs) if self.incorrect_pairs else 0,
            "model_name": self.model_name,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Partial files updated in '{results_dir}' directory")

    def save_results(self, final_line_number: int = None):
        """Save final results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"assessment_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        correct_file = results_dir / f"dict_correct_pairs_{timestamp}.txt"
        with open(correct_file, "w", encoding="utf-8") as f:
            for source, target in self.correct_pairs:
                f.write(f"{source} -> {target}\n")
        
        incorrect_file = results_dir / f"dict_incorrect_pairs_{timestamp}.txt"
        with open(incorrect_file, "w", encoding="utf-8") as f:
            for source, target in self.incorrect_pairs:
                f.write(f"{source} -> {target}\n")
        
        corrected_file = results_dir / f"dict_corrected_pairs_{timestamp}.txt"
        with open(corrected_file, "w", encoding="utf-8") as f:
            for source, corrected_target in self.corrected_pairs:
                f.write(f"{source} -> {corrected_target}\n")
        
        clean_file = results_dir / f"dict_clean_combined_{timestamp}.txt"
        with open(clean_file, "w", encoding="utf-8") as f:
            for source, target in self.correct_pairs:
                f.write(f"{source} -> {target}\n")
            for source, corrected_target in self.corrected_pairs:
                f.write(f"{source} -> {corrected_target}\n")
        
        summary_file = results_dir / f"assessment_summary_{timestamp}.json"
        summary = {
            "total_pairs": len(self.results),
            "final_processed_line": final_line_number,
            "correct_pairs": len(self.correct_pairs),
            "incorrect_pairs": len(self.incorrect_pairs),
            "corrected_pairs": len(self.corrected_pairs),
            "unclear_pairs": len(self.results) - len(self.correct_pairs) - len(self.incorrect_pairs),
            "accuracy_rate": len(self.correct_pairs) / len(self.results) if self.results else 0,
            "correction_rate": len(self.corrected_pairs) / len(self.incorrect_pairs) if self.incorrect_pairs else 0,
            "model_name": self.model_name,
            "timestamp": timestamp
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to '{results_dir}' directory:")
        logger.info(f"  - Detailed results: {results_file.name}")
        logger.info(f"  - Correct pairs: {correct_file.name} ({len(self.correct_pairs)} pairs)")
        logger.info(f"  - Incorrect pairs: {incorrect_file.name} ({len(self.incorrect_pairs)} pairs)")
        logger.info(f"  - Corrected pairs: {corrected_file.name} ({len(self.corrected_pairs)} pairs)")
        logger.info(f"  - Clean combined dict: {clean_file.name} ({len(self.correct_pairs) + len(self.corrected_pairs)} pairs)")
        logger.info(f"  - Summary: {summary_file.name}")

    def show_progress(self):
        """Display current progress status."""
        results_dir = Path("results")
        progress_file = results_dir / "assessment_results_partial.json"
        progress_info_file = results_dir / "progress_info.json"
        
        if not progress_file.exists():
            print("No progress file found. No previous processing detected.")
            return
        
        try:
            if progress_info_file.exists():
                with open(progress_info_file, "r", encoding="utf-8") as f:
                    progress_info = json.load(f)
                    
                print("=== PROGRESS STATUS ===")
                print(f"Last processed line: {progress_info.get('last_processed_line', 'Unknown')}")
                print(f"Total pairs processed: {progress_info.get('total_processed', 'Unknown')}")
                print(f"Last updated: {progress_info.get('last_updated', 'Unknown')}")
                print(f"Model used: {progress_info.get('model_name', 'Unknown')}")
                
                last_line = progress_info.get('last_processed_line')
                if last_line:
                    next_line = last_line + 1
                    print(f"\nTo resume processing, use: --resume-from-line {next_line}")
                
            else:
                with open(progress_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    
                print("=== PROGRESS STATUS ===")
                print(f"Total pairs processed: {len(results)}")
                
                if results:
                    last_result = results[-1]
                    last_line = last_result.get('line_number', len(results))
                    print(f"Last processed line: {last_line}")
                    print(f"Last processed pair: '{last_result.get('source', '?')}' -> '{last_result.get('target', '?')}'")
                    
                    next_line = last_line + 1
                    print(f"\nTo resume processing, use: --resume-from-line {next_line}")
                
                correct = sum(1 for r in results if r.get('verdict') == 'CORRECT')
                incorrect = sum(1 for r in results if r.get('verdict') == 'INCORRECT')
                unclear = len(results) - correct - incorrect
                
                print(f"\nBreakdown:")
                print(f"  - Correct pairs: {correct}")
                print(f"  - Incorrect pairs: {incorrect}")
                print(f"  - Unclear pairs: {unclear}")
                
                if len(results) > 0:
                    print(f"  - Accuracy rate: {correct/len(results)*100:.1f}%")
            
            print("\nPartial result files in 'results/' directory:")
            for partial_file in results_dir.glob("*_partial.*"):
                print(f"  - {partial_file.name}")
                
        except Exception as e:
            print(f"Error reading progress files: {e}")


def main():
    """Main function to run the dictionary assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assess Turkish-English dictionary alignments using Qwen2.5-7b-instruct")
    parser.add_argument("--dict-file", default="dict_llm_5k_lemma_lowercase.txt", 
                       help="Path to dictionary file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for saving progress")
    parser.add_argument("--max-pairs", type=int, default=None,
                       help="Maximum number of pairs to process (for testing)")
    parser.add_argument("--test-run", action="store_true",
                       help="Run on first 20 pairs only for testing")
    parser.add_argument("--resume-from-line", type=int, default=None,
                       help="Resume processing from specific line number (1-indexed)")
    parser.add_argument("--show-progress", action="store_true",
                       help="Show current progress status and exit")
    
    args = parser.parse_args()
    
    if args.show_progress:
        temp_assessor = DictAssessment.__new__(DictAssessment)
        temp_assessor.show_progress()
        return
    
    if args.test_run:
        args.max_pairs = 20
        logger.info("Running in test mode with 20 pairs")
    
    if args.resume_from_line:
        logger.info(f"Resume mode: starting from line {args.resume_from_line}")
    
    assessor = DictAssessment(model_name=args.model)
    
    assessor.process_dictionary(
        dict_file=args.dict_file,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        resume_from_line=args.resume_from_line
    )


if __name__ == "__main__":
    main() 