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
    
import mne
import numpy as np
from src.preprocess import EEGLoader
from src.utils import compare_3d_trajectories

def run_comparison():
    print("--- Comparing Left vs Right Motor Imagery ---")
    
    # 1. 데이터 로드
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    
    # 2. 이벤트(라벨) 추출
    events, event_id = mne.events_from_annotations(raw)
    # T1: Left Hand (보통 ID 2), T2: Right Hand (보통 ID 3)
    # event_id 구성에 따라 다를 수 있으니 확인 필요
    print(f"Event ID Mapping: {event_id}")

    # 3. 라벨별 샘플 구간 찾기
    # 첫 번째로 등장하는 왼손(T1)과 오른손(T2) 시점을 찾음
    t1_start = events[events[:, 2] == event_id['T1']][0][0]
    t2_start = events[events[:, 2] == event_id['T2']][0][0]
    
    # 4. 해당 구간 전처리 (Analytic Signal로 변환)
    # 이벤트 발생 시점부터 약 2초(320 samples 정도, 샘플링 레이트 160Hz 기준) 잘라냄
    duration = 320 
    
    left_hand_raw = raw.copy().crop(tmin=t1_start/raw.info['sfreq'], 
                                    tmax=(t1_start+duration)/raw.info['sfreq'])
    right_hand_raw = raw.copy().crop(tmin=t2_start/raw.info['sfreq'], 
                                     tmax=(t2_start+duration)/raw.info['sfreq'])
    
    z_left, _ = loader.process_to_analytic(left_hand_raw)
    z_right, _ = loader.process_to_analytic(right_hand_raw)
    
    # 5. 비교 시각화
    print("Visualizing comparison...")
    compare_3d_trajectories([z_left, z_right], ['Left Hand', 'Right Hand'], 
                            channel_idx=0, duration=duration)


if __name__ == "__main__":
    # run_research_pipeline()
    run_comparison()