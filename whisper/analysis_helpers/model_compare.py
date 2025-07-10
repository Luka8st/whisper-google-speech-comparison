import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks")

def compare_model_outputs(whisper_csv, google_csv, save_dir="plots"):
    w = pd.read_csv(whisper_csv)
    g = pd.read_csv(google_csv)

    w = w.rename(columns={"wer": "wer_w", "cer": "cer_w", "der": "der_w", "hyp": "hyp_w"})
    g = g.rename(columns={"wer": "wer_g", "cer": "cer_g", "der": "der_g", "hyp": "hyp_g"})

    # in g convert male to m and female to f
    g['gender'] = g['gender'].map({'male': 'm', 'female': 'f'})
    

    #print("Whisper:")
    #print(w.head())
    #print("\nGoogle:")
    #print(g.head())
    df = pd.merge(w, g, on=["file_id", "gender", "ref"])
    #df = pd.merge(w, g, on=["file_id", "ref"])
    print(df.head())
    print(df.columns)
    df["wer_diff"] = df["wer_g"] - df["wer_w"]
    df["cer_diff"] = df["cer_g"] - df["cer_w"]
    df["der_diff"] = df["der_g"] - df["der_w"]

    for metric in ["wer", "cer", "der"]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[f"{metric}_diff"], bins=30, kde=True)
        plt.axvline(0, color="black", linestyle="--")
        plt.title(f"Razlika {metric.upper()} (Google - Whisper)")
        plt.xlabel(f"{metric.upper()} razlika")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_diff_hist.png")
        plt.close()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=df[f"{metric}_w"], y=df[f"{metric}_g"], hue=df["gender"])
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel(f"Whisper {metric.upper()}")
        plt.ylabel(f"Google {metric.upper()}")
        plt.title(f"{metric.upper()} usporedba po isječku")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{metric}_scatter.png")
        plt.close()

    return df