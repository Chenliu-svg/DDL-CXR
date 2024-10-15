"""
This script pickles ehr for ldm and prediction stage.
python pickel_file.py --metadata_path /your/metadata_path --mimic_iv_subjects_dir /your/mimic_iv_subjects_dir
"""

import pickle
import pandas as pd
import numpy as np
import os
from PIL import Image
from ldm.data.dynamic_data import EHR_Pickler
import torchvision.transforms as transforms
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--metadata_path', required=True, help='where the meta csv files are')
    parser.add_argument('--mimic_iv_subjects_dir', required=True, help='The path to the processed mimic-iv subjects files, where the  "\{subject_id\}/episode\{stay_id\}_timeseries.csv" is.')
    
    args = parser.parse_args()

    # pikcle ehr for ldm
    ldm_picler=EHR_Pickler(stage='ldm',metadata_path=args.metadata_path,mimic_iv_subjects_dir=args.mimic_iv_subjects_dir)
    ldm_picler.pickle_ehr()
    
    pred_picler=EHR_Pickler(stage='prediction',metadata_path=args.metadata_path,mimic_iv_subjects_dir=args.mimic_iv_subjects_dir)
    pred_picler.pickle_ehr()
    

