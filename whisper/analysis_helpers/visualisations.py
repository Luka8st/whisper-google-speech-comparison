import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_metrics(csv_path, save_dir="plots"):
    df = pd.read_csv(csv_path)
    metrics = ['wer', 'cer', 'der']
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="model", y=metric, hue="gender")
        plt.title(f"{metric.upper()} po modelu i spolu")
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_by_model_gender.png")
        plt.close()

def summary_stats(csv_path):
    df = pd.read_csv(csv_path)
    return df.groupby(['model', 'gender'])[['wer', 'cer', 'der']].mean().round(3)
