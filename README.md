# whisper-google-speech-comparison

## Struktura projekta

### Google 
- `file_cleanup.py` - Čišćenje .txt datoteka
- `main.py` - Iteriranje kroz .wav i .txt datoteke, pohrana rezultata
- `analysis.py` - Prikaz grafova
- `veprad_audio/` - .wav datoteke
- `veprad_transcripts/` - .txt datoteke

### Whisper

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
4. 
