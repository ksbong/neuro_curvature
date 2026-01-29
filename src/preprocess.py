import mne
import os
import numpy as np
from scipy.signal import hilbert

class EEGLoader:
    def __init__(self, save_path="data/raw"):
        self.save_path = save_path
        # 폴더가 없으면 생성
        os.makedirs(self.save_path, exist_ok=True)

    def fetch_and_load(self, subject=1, runs=[4, 8, 12]):
        """
        PhysioNet의 EEGBCI 데이터를 자동으로 다운로드하고 로드함.
        runs [4, 8, 12]는 보통 Motor Imagery (손/발 상상) 데이터임.
        """
        print(f"--- Fetching EEGBCI Data for Subject {subject} ---")
        # 자동 다운로드 및 경로 반환
        files = mne.datasets.eegbci.load_data(subject, runs, path=self.save_path)
        
        # 데이터 합치기 (Concat)
        raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
        raw = mne.concatenate_raws(raws)
        
        # 채널 이름 표준화 (10-20 시스템)
        mne.datasets.eegbci.standardize(raw)
        raw.set_montage('standard_1020')
        
        return raw

    def process_to_analytic(self, raw, l_freq=0.5, h_freq=50.0):
        """
        필터링 및 Hilbert Transform 수행
        """
        raw.filter(l_freq, h_freq, fir_design='firwin')
        data = raw.get_data()
        analytic_signal = hilbert(data, axis=-1)
        return analytic_signal, raw.info