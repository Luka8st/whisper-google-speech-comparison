import os
import re
import csv
from collections import Counter
from google.cloud import speech
import jiwer

AUDIO_DIR = 'google/veprad_audio/'
TRANSCRIPT_DIR = 'google/veprad_transcripts/cleaned/'
RESULTS_DIR = 'google/results/'
RESULTS_CSV = 'google/evaluation_results.csv'

CHAR_MAP = {
    '{': 'š',
    '~': 'č',
    '^': 'ć',
    '`': 'ž',
    '}': 'đ',
}

CHAR_MAP.update({k.upper(): v.upper() for k, v in CHAR_MAP.items()})
TRANSLATOR = str.maketrans(CHAR_MAP)

def clean_transcript(text):
    text = text.translate(TRANSLATOR)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)

def get_google_transcription(audio_path, result_path):
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    print(f"  -> Google API za {os.path.basename(audio_path)}...")
    client = speech.SpeechClient()
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='hr-HR',
        enable_automatic_punctuation=False
    )
    try:
        response = client.recognize(config=config, audio=audio)
        hypothesis = response.results[0].alternatives[0].transcript.lower() if response.results else ""
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(hypothesis)
        return hypothesis
    except Exception as e:
        print(f"Greška pri transkripciji datoteke {audio_path}: {e}")
        return None

def analyze_and_print_summary(references, hypotheses, label):
    if not references:
        print(f"Nema podataka za sažetak za: {label}")
        return

    print("\n" + "="*50)
    print(f" SAŽETAK REZULTATA ZA: {label.upper()} ")
    print("="*50)
    
    report = jiwer.process_words(references, hypotheses)
    
    total_words_in_reference = report.hits + report.substitutions + report.deletions

    print(f"Ukupna stopa pogreške riječi (WER): {report.wer * 100:.2f}%")
    print(f"Broj supstitucija (S): {report.substitutions}")
    print(f"Broj delecija (D): {report.deletions}")
    print(f"Broj insercija (I): {report.insertions}")
    print(f"Ukupan broj riječi u referenci (N): {total_words_in_reference}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    male_references, male_hypotheses = [], []
    female_references, female_hypotheses = [], []

    try:
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    except FileNotFoundError:
        print(f"Direktorij '{AUDIO_DIR}' nije pronađen. Molimo provjerite putanju.")
        return

    print(f"Pronađeno {len(audio_files)} audio datoteka za obradu.")
    print(f"Detaljni rezultati će biti zapisani u datoteku: '{RESULTS_CSV}'")

    csv_header = ['gender', 'file_id', 'wer', 'cer', 'der', 'ref', 'hyp']

    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        for audio_filename in sorted(audio_files):
            basename = os.path.splitext(audio_filename)[0]
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            transcript_path = os.path.join(TRANSCRIPT_DIR, basename + '.txt')
            result_path = os.path.join(RESULTS_DIR, basename + '.txt')

            if not os.path.exists(transcript_path):
                continue

            print(f"Obrađujem: {audio_filename}...")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                reference_text = clean_transcript(f.read())

            hypothesis_text = get_google_transcription(audio_path, result_path)
            if hypothesis_text is None:
                continue

            word_report = jiwer.process_words(reference_text, hypothesis_text)
            char_report = jiwer.process_characters(reference_text, hypothesis_text)
            
            n_words = word_report.hits + word_report.substitutions + word_report.deletions
            
            wer = word_report.wer
            cer = char_report.cer
            der = word_report.deletions / n_words if n_words > 0 else 0.0
            
            gender = "male" if audio_filename.startswith('m') else "female"
            
            row_data = [
                gender, basename,
                f"{wer:.4f}", f"{cer:.4f}", f"{der:.4f}",
                reference_text, hypothesis_text
            ]
            csv_writer.writerow(row_data)

            if gender == "male":
                male_references.append(reference_text)
                male_hypotheses.append(hypothesis_text)
            else:
                female_references.append(reference_text)
                female_hypotheses.append(hypothesis_text)

    all_references = male_references + female_references
    all_hypotheses = male_hypotheses + female_hypotheses
    
    analyze_and_print_summary(male_references, male_hypotheses, "Muški govornici")
    analyze_and_print_summary(female_references, female_hypotheses, "Ženski govornici")
    analyze_and_print_summary(all_references, all_hypotheses, "Ukupno (svi govornici)")

    print(f"\nObrada završena. Detaljni rezultati su spremljeni u '{RESULTS_CSV}'")

if __name__ == '__main__':
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        print("Greška: Varijabla okruženja 'GOOGLE_APPLICATION_CREDENTIALS' nije postavljena.")
    else:
        main()