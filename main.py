from src.preprocess import EEGLoader
from src.features import GeometryExtractor
from src.encoder import GeometricSpikeEncoder # 추가
from src.utils import plot_complex_trajectory_3d
import torch
import matplotlib.pyplot as plt
import numpy as np


def run_research_pipeline():
    print("--- Starting NeuroCurvature Event-based Pipeline ---")
    
    # 1. 데이터 로드 및 전처리
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    analytic_data, info = loader.process_to_analytic(raw)
    
    # 2. 기하학적 특징 추출 (곡률)
    curvature = GeometryExtractor.calculate_curvature(analytic_data)
    
    # 3. Geometric Spike Encoding
    encoder = GeometricSpikeEncoder(threshold_type='std')
    spikes, threshold = encoder.threshold_encoding(curvature)
    
    # 결과 확인
    n_spikes = torch.sum(spikes).item()
    total_bins = spikes.numel()
    print(f"Spiking Density: {(n_spikes/total_bins)*100:.2f}% ({int(n_spikes)} spikes)")

    # 4. 시각화 (곡률 vs 스파이크)
    # 특정 채널의 시간 흐름에 따른 곡률과 발생한 스파이크를 비교해봐
    ch = 0
    duration = 500
    plt.figure(figsize=(12, 4))
    plt.plot(curvature[ch, :duration], label='Curvature ($\kappa$)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.stem(np.where(spikes[ch, :duration] > 0)[0], 
             [curvature[ch, i] for i in np.where(spikes[ch, :duration] > 0)[0]], 
             'g', markerfmt='go', label='Generated Spikes')
    plt.title(f"Geometric Spike Generation - Channel {ch}")
    plt.legend()
    plt.show()

    # 3D 궤적도 다시 한 번 확인 (이 스파이크들이 '꺾이는 지점'에서 나오는지)
    plot_complex_trajectory_3d(analytic_data, channel_idx=ch, duration=1000)

if __name__ == "__main__":
    run_research_pipeline()