import pandas as pd
import numpy as np
import os

corpus = open("./src/corpus", mode="a")
for filename in os.listdir("./data"):
    with open("./data/"+filename, encoding='ascii', errors='ignore') as book:
        for line in book:
            corpus.write(line)
