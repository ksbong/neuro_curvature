import numpy as np
import torch

class GeometricSpikeEncoder:
    def __init__(self, threshold_type='mean', custom_threshold=None):
        self.threshold_type = threshold_type
        self.custom_threshold = custom_threshold

    def threshold_encoding(self, curvature):
        """
        곡률이 임계값을 넘으면 1(Spike), 아니면 0을 반환함.
        """
        if self.threshold_type == 'mean':
            # 전체 채널/시간의 평균 곡률을 임계값으로 설정
            threshold = np.mean(curvature) * 1.5 
        elif self.threshold_type == 'std':
            # 평균 + 표준편차 기반 임계값
            threshold = np.mean(curvature) + np.std(curvature)
        else:
            threshold = self.custom_threshold if self.custom_threshold else 0.5
        
        print(f"--- Encoding with threshold: {threshold:.4f} ---")
        
        # 스파이크 생성 (1: Spike, 0: Silence)
        spikes = (curvature > threshold).astype(np.float32)
        return torch.from_numpy(spikes), threshold

    def latency_encoding(self, curvature, num_steps=20):
        """
        곡률이 높을수록 시간 윈도우 내에서 '빨리' 스파이크가 발생하게 함. (Latency Encoding)
        """
        # 곡률 값을 0~1 사이로 정규화
        norm_curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-9)
        
        # 높은 곡률일수록 더 짧은 latency(더 빠른 시간 스텝)를 가짐
        # (간단한 구현을 위해 1D 텐서 기준 예시)
        # 실제 SNN 학습 시에는 snntorch.spikegen.latency 기능을 활용하면 더 강력함
        return torch.from_numpy(norm_curvature)