import matplotlib.pyplot as plt
import numpy as np

def plot_complex_trajectory_3d(z, channel_idx=0, duration=1000):
    """
    복소 평면 궤적을 시간축(Z축)과 함께 3차원으로 시각화함.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 데이터 슬라이싱 (너무 길면 그래프가 무거워지니 duration만큼만)
    z_slice = z[channel_idx, :duration]
    real = z_slice.real
    imag = z_slice.imag
    time = np.arange(len(z_slice)) # 시간축
    
    # 3D 선 그래프 그리기
    ax.plot(real, imag, time, label=f'Channel {channel_idx}', alpha=0.8)
    
    # 축 설정
    ax.set_xlabel('Real (Original Signal)')
    ax.set_ylabel('Imaginary (Hilbert)')
    ax.set_zlabel('Time (Steps)')
    ax.set_title(f'3D Spatio-temporal EEG Trajectory - Ch {channel_idx}')
    
    plt.legend()
    plt.show()
    

def compare_3d_trajectories(z_list, labels, channel_idx=0, duration=1000):
    """
    여러 라벨의 복소 궤적을 3차원 상에서 나란히 비교함.
    """
    fig = plt.figure(figsize=(15, 7))
    
    for i, (z, label) in enumerate(zip(z_list, labels)):
        ax = fig.add_subplot(1, len(z_list), i+1, projection='3d')
        
        # 데이터 슬라이싱
        z_slice = z[channel_idx, :duration]
        real, imag = z_slice.real, z_slice.imag
        time = np.arange(len(z_slice))
        
        # 궤적 그리기
        ax.plot(real, imag, time, alpha=0.8, label=label)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_zlabel('Time')
        ax.set_title(f'Trajectory: {label}')
        ax.legend()
        
    plt.tight_layout()
    plt.show()
    
import mne
from src.core.preprocess import EEGLoader
from src.core.geometry import GeometryExtractor
from src.data.encoder import GeometricSpikeEncoder

def run_research_pipeline():
    print("\n--- Phase 1: Basic Research Pipeline ---")
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    analytic_data, _ = loader.process_to_analytic(raw)
    
    curvature = GeometryExtractor.calculate_curvature(analytic_data)
    encoder = GeometricSpikeEncoder(threshold_type='std')
    spikes, threshold = encoder.threshold_encoding(curvature)
    
    plt.figure(figsize=(12, 4))
    plt.plot(curvature[0, :500], label=r'Curvature ($\kappa$)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.legend(); plt.show()

def run_quantitative_comparison():
    print("\n--- Phase 3: Quantitative Analysis ---")
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    events, event_id = mne.events_from_annotations(raw)
    
    sfreq = raw.info['sfreq']
    duration = int(sfreq * 3)
    
    t1_idx = events[events[:, 2] == event_id['T1']][0][0]
    t2_idx = events[events[:, 2] == event_id['T2']][0][0]
    
    z_l, _ = loader.process_to_analytic(raw.copy().crop(tmin=t1_idx/sfreq, tmax=(t1_idx+duration)/sfreq))
    z_r, _ = loader.process_to_analytic(raw.copy().crop(tmin=t2_idx/sfreq, tmax=(t2_idx+duration)/sfreq))
    
    curv_l = GeometryExtractor.calculate_curvature(z_l)
    curv_r = GeometryExtractor.calculate_curvature(z_r)
    
    plt.hist(curv_l.flatten(), bins=100, alpha=0.5, label='Left', color='blue', range=(0, 2))
    plt.hist(curv_r.flatten(), bins=100, alpha=0.5, label='Right', color='red', range=(0, 2))
    plt.title("Curvature Distribution")
    plt.legend(); plt.show()
