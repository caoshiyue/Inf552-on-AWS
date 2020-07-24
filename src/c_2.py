import pandas as pd
import numpy as np
import os

char_set = set(open('./src/corpus').read().lower())
char_set = sorted(list(char_set))
char_2_float = dict()
char_2_int = dict()
int_2_char = dict()
i = 0
for c in char_set:
    char_2_float[c] = i/len(char_set)
    i += 1
i = 0
for c in char_set:
    char_2_int[c] = i
    i += 1
i = 0
for c in char_2_float:
    int_2_char[i] = c
    i += 1
