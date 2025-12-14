#!/usr/bin/env python3
import re
import unicodedata
import sentencepiece as spm
from typing import List, Optional, Dict, Any

def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """
    Normalize Unicode text to a standard form.
    
    Args:
        text: The text to normalize
        form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
    
    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)

def fix_punctuation(text: str, lang: str = 'en') -> str:
    """
    Fix spacing around punctuation based on language-specific rules.
    
    Args:
        text: The text to process
        lang: Language code (default: 'en' for English)
    
    Returns:
        Text with fixed punctuation spacing
    """
    # Common punctuation fixes for most languages
    text = re.sub(r' ([.,!?:;)\]}])', r'\1', text)
    text = re.sub(r'([({\[]) ', r'\1', text)
    
    # Fix quotes based on language
    if lang in ['en', 'de', 'es', 'it']:
        # Fix English-style quotes
        text = re.sub(r' " ', r' "', text)
        text = re.sub(r' " ', r'" ', text)
        text = re.sub(r' \' ', r' \'', text)
        text = re.sub(r' \' ', r'\' ', text)
    elif lang in ['fr']:
        # French-style quotes with spaces
        text = re.sub(r' « ', r' « ', text)
        text = re.sub(r' » ', r' » ', text)
    
    # Fix ellipsis
    text = re.sub(r' \. \. \.', r'...', text)
    text = re.sub(r'\. \. \.', r'...', text)
    
    return text

def fix_numbers_and_dates(text: str) -> str:
    """
    Fix spacing in numbers, dates, and units.
    
    Args:
        text: The text to process
    
    Returns:
        Text with fixed number and date formatting
    """
    # Fix decimal numbers (e.g., "3 . 14" -> "3.14")
    text = re.sub(r'(\d) \. (\d)', r'\1.\2', text)
    
    # Fix thousands separators (e.g., "1 , 000" -> "1,000")
    text = re.sub(r'(\d) , (\d)', r'\1,\2', text)
    
    # Fix date formats (e.g., "01 / 01 / 2023" -> "01/01/2023")
    text = re.sub(r'(\d{1,4}) \/ (\d{1,2}) \/ (\d{1,4})', r'\1/\2/\3', text)
    text = re.sub(r'(\d{1,4}) - (\d{1,2}) - (\d{1,4})', r'\1-\2-\3', text)
    
    # Fix time formats (e.g., "12 : 30" -> "12:30")
    text = re.sub(r'(\d{1,2}) : (\d{2})', r'\1:\2', text)
    
    # Fix units (e.g., "10 kg" should stay as "10 kg", not "10kg")
    text = re.sub(r'(\d) ([a-zA-Z]+)', r'\1 \2', text)
    
    # Fix percentages (e.g., "10 %" -> "10%")
    text = re.sub(r'(\d) %', r'\1%', text)
    
    return text

def fix_named_entities(text: str) -> str:
    """
    Fix common issues with named entities.
    
    Args:
        text: The text to process
    
    Returns:
        Text with fixed named entity formatting
    """
    # Fix common abbreviations (e.g., "U . S . A ." -> "U.S.A.")
    text = re.sub(r'([A-Z]) \. ([A-Z]) \. ([A-Z]) \.', r'\1.\2.\3.', text)
    text = re.sub(r'([A-Z]) \. ([A-Z]) \.', r'\1.\2.', text)
    
    # Fix common titles (e.g., "Mr ." -> "Mr.")
    for title in ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof']:
        text = re.sub(rf'{title} \.', f'{title}.', text)
    
    # Fix common organization suffixes
    for suffix in ['Inc', 'Ltd', 'Corp', 'Co']:
        text = re.sub(rf'{suffix} \.', f'{suffix}.', text)
    
    return text

def detokenize_sentencepiece(sentences: List[str], 
                            lang: str = 'en', 
                            normalize: bool = True,
                            fix_entities: bool = True,
                            fix_numbers: bool = True) -> List[str]:
    """
    Comprehensive detokenization for SentencePiece-tokenized text.
    
    Args:
        sentences: List of tokenized sentences
        lang: Language code for language-specific rules
        normalize: Whether to apply Unicode normalization
        fix_entities: Whether to apply named entity fixes
        fix_numbers: Whether to fix numbers and dates
    
    Returns:
        List of detokenized sentences
    """
    detokenized_sentences = []
    
    for sent in sentences:
        detokenized = sent.replace(' ▁', ' ').replace('▁', ' ').strip()
        detokenized = re.sub(r'\s+', ' ', detokenized)
        detokenized = fix_punctuation(detokenized, lang)
        
        if fix_numbers:
            detokenized = fix_numbers_and_dates(detokenized)
        
        if fix_entities:
            detokenized = fix_named_entities(detokenized)
        
        if normalize:
            detokenized = normalize_unicode(detokenized)
        
        detokenized_sentences.append(detokenized)
    
    return detokenized_sentences

def detokenize_bpe_with_model(sentences: List[str], model_path: str) -> List[str]:
    """
    Detokenize BPE-tokenized sentences using the SentencePiece model.
    
    Args:
        sentences: List of tokenized sentences
        model_path: Path to the SentencePiece model
    
    Returns:
        List of detokenized sentences
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    detokenized_sentences = []
    for sent in sentences:
        detokenized = sp.decode(sent.split())
        detokenized_sentences.append(detokenized)
    
    return detokenized_sentences

def detokenize_with_fallback(sentences: List[str], 
                           model_path: Optional[str] = None,
                           lang: str = 'en',
                           normalize: bool = True) -> List[str]:
    """
    Detokenize sentences with fallback mechanism.
    First tries using the SentencePiece model if provided,
    then falls back to manual detokenization if that fails.
    
    Args:
        sentences: List of tokenized sentences
        model_path: Path to the SentencePiece model (optional)
        lang: Language code for language-specific rules
        normalize: Whether to apply Unicode normalization
    
    Returns:
        List of detokenized sentences
    """
    if model_path:
        try:
            return detokenize_bpe_with_model(sentences, model_path)
        except Exception:
            pass
    
    return detokenize_sentencepiece(sentences, lang, normalize)

def check_detokenization_consistency(text: str, detokenized_text: str) -> Dict[str, Any]:
    """
    Check if detokenization was performed consistently.
    
    Args:
        text: Original tokenized text
        detokenized_text: Detokenized text
    
    Returns:
        Dictionary with consistency metrics
    """
    sp_tokens_remaining = len(re.findall(r'▁', detokenized_text))
    
    orig_alphanum = ''.join(c for c in text if c.isalnum())
    detok_alphanum = ''.join(c for c in detokenized_text if c.isalnum())
    content_preserved = orig_alphanum == detok_alphanum
    
    double_spaces = len(re.findall(r'  +', detokenized_text))
    bad_punctuation = len(re.findall(r' [.,!?:;]', detokenized_text))
    
    return {
        'sp_tokens_remaining': sp_tokens_remaining,
        'content_preserved': content_preserved,
        'double_spaces': double_spaces,
        'bad_punctuation': bad_punctuation,
        'is_consistent': (sp_tokens_remaining == 0 and 
                         content_preserved and 
                         double_spaces == 0 and 
                         bad_punctuation == 0)
    } 