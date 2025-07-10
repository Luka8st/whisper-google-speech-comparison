import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

hr_diacritics = ['č','ć','š','ž','đ','Č','Ć','Š','Ž','Đ']

def diacritic_confusion(csv_path, save_path="diacritic_heatmap.png"):
    df = pd.read_csv(csv_path)
    matrix = {d: Counter() for d in hr_diacritics}

    for _, row in df.iterrows():
        ref = row['ref']
        hyp = row['hyp']
        for i, c in enumerate(ref):
            if c in hr_diacritics:
                wrong = hyp[i] if i < len(hyp) else "_"
                matrix[c][wrong] += 1

    labels = sorted(hr_diacritics)
    data = np.zeros((len(labels), len(labels)))
    for i, d1 in enumerate(labels):
        for j, d2 in enumerate(labels):
            data[i, j] = matrix[d1][d2]

    dfm = pd.DataFrame(data, index=labels, columns=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(dfm, annot=True, fmt=".0f", cmap="Reds")
    plt.title("Zamjene dijakritika (Ref → Pogrešan)")
    plt.xlabel("Pogrešan znak")
    plt.ylabel("Ispravan znak")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
