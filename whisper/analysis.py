import analysis_helpers.diacritics_hm, analysis_helpers.visualisations, analysis_helpers.model_compare
import os

os.makedirs("whisper\\test\\whisper_outputs\\analysis_plots", exist_ok=True)
os.makedirs("google\\analysis_plots", exist_ok=True)
os.makedirs("comparison", exist_ok=True)

analysis_helpers.diacritics_hm.diacritic_confusion("whisper\\test\\whisper_outputs\\metrics.csv", "whisper\\test\\whisper_outputs\\analysis_plots\\diacritic_heatmap.png")
analysis_helpers.visualisations.plot_metrics("whisper\\test\\whisper_outputs\\metrics.csv", "whisper\\test\\whisper_outputs\\analysis_plots")

analysis_helpers.model_compare.compare_model_outputs("whisper\\test\whisper_outputs\metrics.csv", "google\evaluation_results.csv", "comparison")