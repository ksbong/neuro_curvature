import torch
import numpy as np

class MultiFeatureEncoder:
    def __init__(self, num_neurons_curv=5, eeg_threshold=1.0):
        self.num_neurons_curv = num_neurons_curv
        # 곡률 분포(Mean 0.04, 95% 0.13)에 최적화된 임계값
        self.curv_thresholds = torch.tensor([0.02, 0.04, 0.06, 0.09, 0.13])
        self.eeg_threshold = eeg_threshold

    def encode(self, curvature_log, raw_eeg):
        if not isinstance(curvature_log, torch.Tensor):
            curvature_log = torch.from_numpy(curvature_log).float()
        if not isinstance(raw_eeg, torch.Tensor):
            raw_eeg = torch.from_numpy(raw_eeg).float()

        # 1. 곡률 인코딩 (64채널 * 5개 임계값 = 320차원)
        curv_spikes = []
        for v in self.curv_thresholds:
            curv_spikes.append((curvature_log > v).float())
        curv_encoded = torch.cat(curv_spikes, dim=0)

        # 2. 원본 EEG 인코딩 (진폭 기반 64차원)
        eeg_encoded = (torch.abs(raw_eeg) > self.eeg_threshold).float()

        # 3. 데이터 결합 (총 384차원)
        return torch.cat([curv_encoded, eeg_encoded], dim=0)