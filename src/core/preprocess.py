import mne
import os
import numpy as np
from scipy.signal import hilbert

class EEGLoader:
    def __init__(self, save_path="data/raw"):
        self.save_path = save_path
        # 폴더가 없으면 생성
        os.makedirs(self.save_path, exist_ok=True)

    # src/core/preprocess.py

    def fetch_and_load(self, subjects=None):
        """
        여러 피험자의 데이터를 로드하여 합침.
        subjects: [1, 2, 3, 4, 5] 식의 리스트
        """
        if subjects is None:
            subjects = [1]
            
        all_raws = []
        for sub in subjects:
            print(f"--- Fetching EEGBCI Data for Subject {sub} ---")
            # runs 4, 8, 12 (Motor Imagery: Left/Right Hand)
            paths = mne.datasets.eegbci.load_data(sub, [4, 8, 12])
            raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
            all_raws.extend(raws)
        
        # 여러 세션을 하나로 합침
        raw = mne.concatenate_raws(all_raws)
        return raw

    def process_to_analytic(self, raw, l_freq=0.5, h_freq=50.0):
        """
        필터링 및 Hilbert Transform 수행
        """
        raw.filter(l_freq, h_freq, fir_design='firwin')
        data = raw.get_data()
        analytic_signal = hilbert(data, axis=-1)
        return analytic_signal, raw.info