#!/usr/bin/env python3
import os
import sys

def detokenize(text):
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    text = text.replace(' :', ':')
    text = text.replace(' ;', ';')
    return text

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    detokenized_lines = [detokenize(line.strip()) for line in lines]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in detokenized_lines:
            f.write(line + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python detokenize.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_file(input_file, output_file) 