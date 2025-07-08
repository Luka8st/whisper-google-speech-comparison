import os
import whisper
import pandas as pd
import numpy as np
from jiwer import wer, cer, ToLowerCase, RemovePunctuation, Compose
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns
import re
import string

def clean_for_wer(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = text.casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

MALE_TXT_DIR   = r".\whisper\test\testtxtM\cleaned"
MALE_WAV_DIR   = r".\whisper\test\testwavM"
FEMALE_TXT_DIR = r".\whisper\test\testtxtF\cleaned"
FEMALE_WAV_DIR = r".\whisper\test\testwavF"

MODELS     = ["small", "large"]
OUTPUT_DIR = r".\whisper\test\whisper_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_ids(txt_dir):
    return [os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith('.txt')]

male_ids   = get_ids(MALE_TXT_DIR)
female_ids = get_ids(FEMALE_TXT_DIR)
file_list  = [('m', fid) for fid in male_ids] + [('f', fid) for fid in female_ids]

diac_chars = ['č','ć','š','ž','đ','Č','Ć','Š','Ž','Đ']

results = []
for model_name in tqdm(MODELS, desc="Models"):
    model = whisper.load_model(model_name)
    for gender, fid in tqdm(file_list, desc=f"{model_name}", leave=False):
        wav_dir = MALE_WAV_DIR   if gender=='m' else FEMALE_WAV_DIR
        txt_dir = MALE_TXT_DIR   if gender=='m' else FEMALE_TXT_DIR
        wav_path = os.path.join(wav_dir, f"{fid}.wav")
        txt_path = os.path.join(txt_dir, f"{fid}.txt")
        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            continue

        out = model.transcribe(wav_path, language='hr')
        hyp = out['text'].strip()

        with open(txt_path, encoding='utf-8') as f:
            ref = f.read().strip()

        cr = clean_for_wer(ref)
        ch = clean_for_wer(hyp)
        
        w = wer(cr, ch)
        c = cer(cr, ch)
        
        total_diacs = sum(ref.count(ch) for ch in diac_chars)
        wrong = sum(
            1 for i, r in enumerate(ref)
            if r in diac_chars and (i >= len(hyp) or hyp[i].lower() != r.lower())
        )
        der = wrong / total_diacs if total_diacs > 0 else np.nan

        results.append({
            'model':   model_name,
            'gender':  gender,
            'file_id': fid,
            'wer':     w,
            'cer':     c,
            'der':     der,
            'ref':     ref,
            'hyp':     hyp
        })


import matplotlib.pyplot as plt

letters = ['č','ć','š','ž','đ']
total = {m:{ch:0 for ch in letters} for m in MODELS}
errors = {m:{ch:0 for ch in letters} for m in MODELS}

for rec in results:
    model = rec['model']
    ref, hyp = rec['ref'], rec['hyp']
    for ch in letters:
        cnt = ref.lower().count(ch)
        total[model][ch] += cnt
        for i, r in enumerate(ref.lower()):
            if r == ch:
                if i >= len(hyp) or hyp.lower()[i] != ch:
                    errors[model][ch] += 1

for model in MODELS:
    rates = [ errors[model][ch]/total[model][ch] 
              if total[model][ch]>0 else 0
              for ch in letters ]
    plt.figure()
    plt.bar(letters, rates)
    plt.title(f"DER po slovu  –  {model}")
    plt.ylabel("Error rate")
    plt.savefig(os.path.join(OUTPUT_DIR, f"der_per_letter_{model}.png"))
    plt.show()

df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nMetrics saved to: {csv_path}")
print(df.head())

for gender, label in [('m', 'Muški'), ('f', 'Ženski')]:
    fig, ax = plt.subplots()
    data = [ df[(df['model']==m) & (df['gender']==gender)]['wer'] for m in MODELS ]
    ax.boxplot(data, labels=MODELS)
    ax.set_title(f"WER - {label}")
    ax.set_xlabel("Model")
    ax.set_ylabel("WER")
    fig.savefig(os.path.join(OUTPUT_DIR, f"wer_boxplot_{gender}.png"))
    plt.show()
    
der_summary = df.groupby('model')['der'].mean()
fig, ax = plt.subplots()
ax.bar(der_summary.index, der_summary.values)
ax.set_title("Prosječni DER po modelu")
ax.set_ylabel("DER")
fig.savefig(os.path.join(OUTPUT_DIR, "avg_der_per_model.png"))
plt.show()

stats = df.groupby(['model','gender'])[['wer','cer','der']].agg(['mean','median','std'])
print("=== Agregirane statistike ===")
print(stats, "\n")

for gender,label in [('m','Muški'), ('f','Ženski')]:
    w_small = df[(df['gender']==gender)&(df['model']=='small')]['wer']
    w_large = df[(df['gender']==gender)&(df['model']=='large')]['wer']
    stat, p = wilcoxon(w_small, w_large)
    print(f"Wilcoxon WER {label}: small vs large → p = {p:.3f}")

for model in df['model'].unique():
    w_m = df[(df['model']==model)&(df['gender']=='m')]['wer']
    w_f = df[(df['model']==model)&(df['gender']=='f')]['wer']
    stat, p = mannwhitneyu(w_m, w_f, alternative='two-sided')
    print(f"Mann–Whitney WER small vs large za model {model}: p = {p:.3f}")

from tqdm import tqdm
from collections import Counter

letters = ['č','ć','š','ž','đ']
der_per_letter = {m: {ch: 0 for ch in letters} for m in MODELS}
total_per_letter = {ch: 0 for ch in letters}

for rec in tqdm(results, desc="Counting DER per letter"):
    model = rec['model']
    ref = rec['ref'].lower()
    hyp = rec['hyp'].lower()
    for ch in letters:
        total_per_letter[ch] += ref.count(ch)
        der_per_letter[model][ch] += sum(
            1 for i, r in enumerate(ref)
            if r == ch and (i >= len(hyp) or hyp[i] != ch)
        )

for model in tqdm(MODELS, desc="Plotting DER bars"):
    rates = [
        der_per_letter[model][ch] / total_per_letter[ch]
        if total_per_letter[ch] > 0 else 0
        for ch in letters
    ]
    plt.figure()
    plt.bar(letters, rates)
    plt.title(f"DER po slovu – {model}")
    plt.ylabel("Error rate")
    plt.savefig(os.path.join(OUTPUT_DIR, f"der_per_letter_{model}.png"))
    plt.show()

tqdm.write("Plotting violinplot for WER distribution…")
plt.figure()
sns.violinplot(data=df, x='model', y='wer', hue='gender', split=True)
plt.title("Distribucija WER po modelu i spolu")
plt.savefig(os.path.join(OUTPUT_DIR, "wer_violinplot.png"))
plt.show()

tqdm.write("Computing confusion pairs for diacritics…")
conf = {m: Counter() for m in MODELS}
for rec in tqdm(results, desc="Counting confusion pairs"):
    model = rec['model']
    r, h = rec['ref'].lower(), rec['hyp'].lower()
    for i, char in enumerate(r):
        if char in letters:
            hyp_char = h[i] if i < len(h) else None
            conf[model][(char, hyp_char)] += 1

print("Confusion č/ć/š/ž/đ za model large:")
for (r, h), cnt in conf['large'].most_common(10):
    print(f"  {r} → {h} : {cnt} puta")

tqdm.write("Plotting WER boxplot across models…")
plt.figure()
sns.boxplot(data=df, x='model', y='wer')
plt.title("WER usporedba svih modela")
plt.savefig(os.path.join(OUTPUT_DIR, "wer_boxplot_all_models.png"))
plt.show()