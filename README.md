# whisper-google-speech-comparison

## Struktura projekta

Projekt je podijeljen na dva dijela (Google i Whisper), svaki od njih ima pripadajući direktorij.

### Google 
- `file_cleanup.py` - Čišćenje .txt datoteka
- `main.py` - Iteriranje kroz .wav i .txt datoteke, pohrana rezultata
- `analysis.py` - Prikaz grafova
- `veprad_audio/` - .wav datoteke
- `veprad_transcripts/` - .txt datoteke

### Whisper
- `file_cleanup.py` - Čišćenje .txt datoteka
- `main.py` - Iteriranje kroz .wav i .txt datoteke, pohrana rezultata
- `analysis.py` - Prikaz grafova i usporeedba s Googleovim modelom
- `test/` - Sadrži sve .wav i .txt datoteke
- `analysis_helpers/` - Pomoćne funkcije za statističku analizu

## Pokretanje koda

### Google
1. **Kreirajte i preuzmite API ključ**
   Na stranici [Google cloud console](https://console.cloud.google.com/) kreirajte API ključ za speech-to-text API i preuzmite ga u JSON formatu
2. **Postavite API ključ kao varijablu okruženja**
   U direktoriju gdje se projekt nalazi, pokrenite sljedeću naredbu u terminalu (Napomena: prilagodite putanju do JSON datoteke)
   ```sh
      $env:GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
   ```
3. **Pokrenite `file_cleanup.py`**
   Ovaj kod 'čisti' tekstualne datoteke i pohranjuje ih u `veprad_transcripts/cleaned/`
4. **Pokrenite `main.py`**
   Prolazi kroz sve datoteke, evaluira model nad njima i pohranjuje rezultate 
5. **Pokrenite `analysis.py`**
   Ovaj kod radi statističku analizu i generira dijagrame na temelju rezultata
   - DER po svim dijakritičkim znakovima
   - Boxplot dijagram za WER po spolu i govorniku
   - *Violin plot* za prikaz distribucije WER-a po spolu

### Whisper
1. **Pokrenite `file_cleanup.py`**
   Ovaj kod 'čisti' tekstualne datoteke i pohranjuje očišćene datoteke
2. **Pokrenite `main.py`**
   Ovaj kod evaluira Whisperove modele nad svim .wav i .txt datotekama i pohranjuje rezultate u `test/whisper_outputs/`
3. **Pokrenite `analysis.py`**
   Vrši statističku analizu, generira dijagrame i uspoređuje rezultate s Googleovim modelom
   - Dijagrami za DER po svim dijakritičkim znakovima
   - Grafovi se pohranjuju u `whisper_outputs/analysis_plots/`
      - Boxplot dijagrami za usporedbu CER, DER i WER po spolu i modelu 
   - Rezultati usporedbe pohranjuju se unutar *root* direktorija u `comparison/`
      - Pohranjuju se *difference histogram* i *scatter plot* za CER, DER i WER 
