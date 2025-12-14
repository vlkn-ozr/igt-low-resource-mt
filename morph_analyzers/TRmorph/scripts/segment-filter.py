#!/usr/bin/env python3
"""
Script to filter segmentation results, keeping only segments that match the original word.
"""

import sys
import codecs

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stdin = codecs.getreader('utf8')(sys.stdin)

for line in sys.stdin:
    if len(line) == 1:
        continue
    t = line.rstrip().split("\t")
    if len(t) != 2:
        continue
    word, seg = t
    tmp = seg.replace("-", "")
    if tmp.lower() == word.lower():
        print(word, '\t', seg)
