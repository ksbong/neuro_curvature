# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class GeometricEEGDataset(Dataset):
    def __init__(self, spikes, events, event_id, window_size=160):
        """
        spikes: (Channels, Time) 형태의 스파이크 텐서
        events: MNE에서 추출한 이벤트 배열
        event_id: {'T1': 2, 'T2': 3} 식의 매핑
        window_size: 160 (160Hz 기준 1초)
        """
        self.samples = []
        self.labels = []
        
        # T1(Left)과 T2(Right) 이벤트만 필터링 (T0 제외)
        target_ids = {event_id['T1']: 0, event_id['T2']: 1}
        
        for onset, _, label in events:
            if label in target_ids:
                # 이벤트 발생 시점부터 window_size만큼 잘라냄
                if onset + window_size < spikes.shape[1]:
                    window = spikes[:, onset:onset+window_size]
                    # SNN 입력을 위해 (Time, Channels) 형태로 변환
                    self.samples.append(window.t()) 
                    self.labels.append(target_ids[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]