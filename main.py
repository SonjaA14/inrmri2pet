from datetime import datetime
import json
import os
import pandas as pd
from train import train

OUTPUT_DIR = "outputs/"
BASE_PATH = "processed/"
LEARNING_RATE = 1e-5
MAX_EPOCHS = 75
ABLATION_SETS = {
    "full": ['t1', 't2', 'adc', 'ttp', 'Ktrans', 've', 'vp'],
    # leave-one-out
    "minus_t1":  ['t2', 'adc', 'ttp', 'Ktrans', 've', 'vp'],
    "minus_t2":  ['t1', 'adc', 'ttp', 'Ktrans', 've', 'vp'],
    "minus_adc": ['t1', 't2', 'ttp', 'Ktrans', 've', 'vp'],
    "minus_ttp": ['t1', 't2', 'adc', 'Ktrans', 've', 'vp'],
    "minus_Ktrans": ['t1', 't2', 'adc', 'ttp', 've', 'vp'],
    "minus_ve": ['t1', 't2', 'adc', 'ttp', 'Ktrans', 'vp'],
    "minus_vp": ['t1', 't2', 'adc', 'ttp', 'Ktrans', 've'],
    # group ablation
    "minus_structural": ['adc', 'ttp', 'Ktrans', 've', 'vp'],
    "minus_inr_series": ['t1', 't2', 'adc', 'ttp'],
    "minus_dynamic": ['t1', 't2', 'adc']
}

PATIENTS = ["mp_0008", "mp_0032", "mp_0038", "mp_0047", "mp_0057", "mp_0058", "mp_0064", 
            "mp_0065", "mp_0080", "mp_0096", "mp_0099", "mp_0100", "mp_0106", "mp_0112", "mp_0113"]

CSV_PATH = os.path.join(OUTPUT_DIR, f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if __name__ == "__main__":
    with open("./paths.json", "r") as f:
        paths = json.load(f)

    for PATIENT in PATIENTS:
        # check if patient has all required paths ending on nii.gz or .mha, skip if any are missing
        required_paths = ['suv', 't1', 't2', 'adc', 'ct', 'ddf', 'ttp', 'inr']
        if not all(key in paths[PATIENT] and paths[PATIENT][key] and 
                   (paths[PATIENT][key].endswith('.nii.gz') or paths[PATIENT][key].endswith('.mha')) 
                   for key in required_paths):
            print(f"Skipping patient {PATIENT} due to missing or incorrect paths.")
            continue

        print(f"\n=== Starting training for patient {PATIENT} ===")
        recompute_data = True
        for name, feat in ABLATION_SETS.items():
            print(f"\n\n=== Starting ablation set: {name} ===")
            print(f"    Features: {feat}")
            result = train(
                recompute_data=recompute_data,
                suv_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['suv']),
                t1_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['t1']),
                t2_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['t2']),
                adc_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['adc']),
                ct_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['ct']),
                ddf_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['ddf']),
                ttp_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['ttp']),
                inr_path=os.path.join(BASE_PATH, PATIENT, paths[PATIENT]['inr']),
                patient_id=PATIENT,
                output_dir=os.path.join(OUTPUT_DIR, PATIENT),
                epochs=MAX_EPOCHS,
                learning_rate=LEARNING_RATE,
                features=feat,
                feature_set_name=name
            )
            row = pd.DataFrame([result])
            row.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
            recompute_data = False  # Only preprocess once per patient, reuse for all ablation sets
        