#!/usr/bin/env python3
"""
Utility script to extract word alignments from model output.
Demonstrates how to parse the structured alignment format.
"""

import re
from typing import List, Dict, Tuple

def extract_alignments_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract word alignments from text containing alignment tags.
    
    Args:
        text: Raw model output containing <alignments>...</alignments>
        
    Returns:
        List of (turkish_lemma, english_lemma) tuples
    """
    alignments = []
    
    pattern = r'<alignments>(.*?)</alignments>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        alignment_content = match.group(1).strip()
    else:
        if '<alignments>' in text:
            start_idx = text.find('<alignments>') + len('<alignments>')
            alignment_content = text[start_idx:].strip()
        else:
            alignment_content = text
    
    for line in alignment_content.split('\n'):
        line = line.strip()
        if ' - ' in line:
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                turkish = parts[0].strip()
                english = parts[1].strip()
                if turkish and english:
                    alignments.append((turkish, english))
    
    return alignments

def extract_alignments_as_dict(text: str) -> Dict[str, str]:
    """
    Extract alignments as a dictionary for easy lookup.
    
    Args:
        text: Raw model output
        
    Returns:
        Dictionary mapping turkish_lemma -> english_lemma
    """
    alignments = extract_alignments_from_text(text)
    return {turkish: english for turkish, english in alignments}

def validate_alignment_output(text: str) -> Dict[str, any]:
    """
    Validate the format of alignment output.
    
    Args:
        text: Raw model output
        
    Returns:
        Validation results dictionary
    """
    has_start_tag = '<alignments>' in text
    has_end_tag = '</alignments>' in text
    has_both_tags = has_start_tag and has_end_tag
    
    alignments = extract_alignments_from_text(text)
    
    return {
        'has_alignment_tags': has_both_tags,
        'has_start_tag': has_start_tag,
        'has_end_tag': has_end_tag,
        'num_alignments': len(alignments),
        'alignments': alignments,
        'is_valid_format': has_both_tags and len(alignments) > 0
    }

def format_alignments_for_display(alignments: List[Tuple[str, str]]) -> str:
    """
    Format alignments for human-readable display.
    
    Args:
        alignments: List of (turkish, english) tuples
        
    Returns:
        Formatted string
    """
    if not alignments:
        return "No alignments found."
    
    lines = []
    max_turkish_len = max(len(turkish) for turkish, _ in alignments)
    
    for turkish, english in alignments:
        lines.append(f"{turkish:<{max_turkish_len}} -> {english}")
    
    return '\n'.join(lines)

if __name__ == "__main__":
    sample_output = """### Word alignments:
<alignments>
öğrenci - student
dün - yesterday
kütüphane - library
sessiz - quiet
çalış - work
</alignments>"""
    
    print("Sample model output:")
    print(sample_output)
    print("\n" + "="*50)
    
    alignments = extract_alignments_from_text(sample_output)
    print(f"\nExtracted {len(alignments)} alignments:")
    print(format_alignments_for_display(alignments))
    
    alignment_dict = extract_alignments_as_dict(sample_output)
    print(f"\nAs dictionary: {alignment_dict}")
    
    validation = validate_alignment_output(sample_output)
    print(f"\nValidation results:")
    print(f"  Valid format: {validation['is_valid_format']}")
    print(f"  Has tags: {validation['has_alignment_tags']}")
    print(f"  Number of alignments: {validation['num_alignments']}") 