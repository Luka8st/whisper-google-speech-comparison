import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from collections import Counter
from tqdm import tqdm
import jiwer 

RESULTS_CSV = 'evaluation_results.csv'
OUTPUT_DIR = 'analysis_plots/'

def perform_statistical_analysis(df):
    print("\n" + "="*50)
    print(" STATISTIČKA ANALIZA")
    print("="*50)

    stats = df.groupby('gender')[['wer', 'cer', 'der']].agg(['mean', 'median', 'std'])
    print("=== Agregirane statistike (srednja vrijednost, medijan, std. dev.) ===")
    print(stats, "\n")

    min_wer = df['wer'].min()
    max_wer = df['wer'].max()
    iqr_wer = df['wer'].quantile(0.75) - df['wer'].quantile(0.25)
    print(f"Min WER: {min_wer:.2%}, Max WER: {max_wer:.2%}, IQR WER: {iqr_wer:.2%}\n")
    
    overall_avg_der = df['der'].mean()
    print(f"Ukupni prosječni DER (Deletion Error Rate): {overall_avg_der:.2%}\n")

    w_male = df[df['gender'] == 'male']['wer']
    w_female = df[df['gender'] == 'female']['wer']
    
    if not w_male.empty and not w_female.empty:
        stat, p = mannwhitneyu(w_male, w_female, alternative='two-sided')
        print(f"=== Mann–Whitney U test (WER muški vs. ženski) ===")
        print(f"P-vrijednost: {p:.4f}")
        if p < 0.05:
            print("Zaključak: Postoji statistički značajna razlika u WER-u između spolova.")
        else:
            print("Zaključak: Nema statistički značajne razlike u WER-u između spolova.")

def analyze_word_errors(df):
    print("\n" + "="*50)
    print(" ANALIZA NAJČEŠĆIH POGREŠAKA U RIJEČIMA")
    print("="*50)

    references = df['ref'].astype(str).tolist()
    hypotheses = df['hyp'].astype(str).tolist()

    report = jiwer.process_words(references, hypotheses)

    word_error_counter = Counter()
    references_as_words = [s.split() for s in references]
    
    for alignment_chunks, ref_words in zip(report.alignments, references_as_words):
        for chunk in alignment_chunks:
            if chunk.type == 'substitute':
                ref_segment_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                word_error_counter.update(ref_segment_words)
            elif chunk.type == 'delete':
                deleted_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                word_error_counter.update(deleted_words)

    print("\n=== Top 20 riječi na kojima sustav najčešće griješi ===")
    print("(Broj puta koliko je riječ bila supstituirana ili izbrisana)")
    for word, count in word_error_counter.most_common(20):
        print(f"  '{word}': {count} puta")


def analyze_diacritics(df):
    print("\n" + "="*50)
    print(" ANALIZA GREŠAKA NA DIJAKRITICIMA")
    print("="*50)

    letters = ['č', 'ć', 'š', 'ž', 'đ']
    total_counts = {ch: 0 for ch in letters}
    error_counts = {ch: 0 for ch in letters}
    confusion_pairs = Counter()

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analiziram dijakritike"):
        ref = str(row['ref'])
        hyp = str(row['hyp'])
        
        for i, ref_char in enumerate(ref):
            if ref_char in letters:
                total_counts[ref_char] += 1
                hyp_char = hyp[i] if i < len(hyp) else None
                
                if ref_char != hyp_char:
                    error_counts[ref_char] += 1
                    confusion_pairs[(ref_char, hyp_char)] += 1
    
    der_rates = {ch: (error_counts[ch] / total_counts[ch] if total_counts[ch] > 0 else 0) for ch in letters}
    print("\n=== Stopa greške po dijakritičkom znaku (DER per letter) ===")
    for ch, rate in der_rates.items():
        print(f"  Slovo '{ch}': {rate:.2%}")

    plt.figure(figsize=(8, 5))
    plt.bar(der_rates.keys(), der_rates.values())
    plt.title("Stopa greške po dijakritiku (DER)")
    plt.ylabel("Stopa greške (Error Rate)")
    plt.savefig(os.path.join(OUTPUT_DIR, "der_per_letter.png"))
    plt.show()

    print("\n=== Najčešći parovi zabune za dijakritike (Referenca -> Hipoteza) ===")
    for (ref_c, hyp_c), count in confusion_pairs.most_common(15):
        print(f"  {ref_c} -> {hyp_c if hyp_c is not None else 'BRISANJE'}: {count} puta")

def create_visualizations(df):
    print("\n" + "="*50)
    print(" STVARANJE VIZUALIZACIJA")
    print("="*50)
    
    print("Violinplot za distribuciju WER-a...")
    print(df.head()) 
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, y='wer', hue='gender', split=True)
    plt.title("Distribucija WER-a po spolu")
    plt.ylabel("Stopa pogreške riječi (WER)")
    plt.xlabel("Spol")
    plt.savefig(os.path.join(OUTPUT_DIR, "wer_violinplot_gender.png"))
    plt.show()

    print("Boxplot za usporedbu WER-a...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='gender', y='wer')
    plt.title("Usporedba WER-a po spolu")
    plt.ylabel("Stopa pogreške riječi (WER)")
    plt.xlabel("Spol")
    plt.savefig(os.path.join(OUTPUT_DIR, "wer_boxplot_gender.png"))
    plt.show()

    print("Boxplot za WER za sve rečenice...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y='wer')
    plt.title("Boxplot WER-a za sve rečenice")
    plt.ylabel("Stopa pogreške riječi (WER)")
    plt.xlabel("Sve rečenice")
    plt.savefig(os.path.join(OUTPUT_DIR, "wer_boxplot_all_sentences.png"))
    plt.show()
    
def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Greška: Datoteka '{RESULTS_CSV}' nije pronađena.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Učitavam rezultate iz '{RESULTS_CSV}'...")
    df = pd.read_csv(RESULTS_CSV)
    
    perform_statistical_analysis(df)
    analyze_word_errors(df)
    analyze_diacritics(df)
    create_visualizations(df)

    print(f"\nAnaliza završena. Svi grafovi su spremljeni u direktorij '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()