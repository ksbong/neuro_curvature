import mne
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 리팩토링된 src 구조에서 임포트 (image_435ddb 에러 해결을 위해 경로 확인)
from src.core.preprocess import EEGLoader
from src.core.geometry import GeometryExtractor
from src.data.encoder import GeometricSpikeEncoder
from src.data.dataset import GeometricEEGDataset  # 새로 만든 데이터셋 클래스
from src.models.snn import GeometricSNN           # spk2 에러가 해결된 모델
from src.utils import plot_complex_trajectory_3d, compare_3d_trajectories

# 1. 초기 연구 파이프라인 (기본 흐름 확인용)
def run_research_pipeline():
    print("\n--- Phase 1: Basic Research Pipeline ---")
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    analytic_data, _ = loader.process_to_analytic(raw)
    
    curvature = GeometryExtractor.calculate_curvature(analytic_data)
    encoder = GeometricSpikeEncoder(threshold_type='std')
    spikes, threshold = encoder.threshold_encoding(curvature)
    
    print(f"Spiking Density: {torch.mean(spikes)*100:.2f}%")
    
    # 시각화 (LaTeX 경고 방지를 위해 r 붙임)
    plt.figure(figsize=(12, 4))
    plt.plot(curvature[0, :500], label=r'Curvature ($\kappa$)')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.legend(); plt.show()
    
    plot_complex_trajectory_3d(analytic_data, channel_idx=0, duration=1000)

# 2. 3D 궤적 비교 (시각적 반전 확인용)
def run_comparison():
    print("\n--- Phase 2: 3D Trajectory Comparison (Left vs Right) ---")
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    events, event_id = mne.events_from_annotations(raw)
    
    sfreq = raw.info['sfreq']
    t1_start = events[events[:, 2] == event_id['T1']][0][0]
    t2_start = events[events[:, 2] == event_id['T2']][0][0]
    
    duration = 320 
    z_left, _ = loader.process_to_analytic(raw.copy().crop(tmin=t1_start/sfreq, tmax=(t1_start+duration)/sfreq))
    z_right, _ = loader.process_to_analytic(raw.copy().crop(tmin=t2_start/sfreq, tmax=(t2_start+duration)/sfreq))
    
    compare_3d_trajectories([z_left, z_right], ['Left Hand', 'Right Hand'], channel_idx=0)

# 3. 정량적 수치 비교 (0.13%의 기적 확인용)
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
    
    combined_mean = (np.mean(curv_l) + np.mean(curv_r)) / 2
    encoder = GeometricSpikeEncoder(threshold_type='custom', custom_threshold=combined_mean * 1.5)
    
    spks_l, _ = encoder.threshold_encoding(curv_l)
    spks_r, _ = encoder.threshold_encoding(curv_r)
    
    print(f"\n[Result] Spiking Density - Left: {torch.mean(spks_l)*100:.2f}%, Right: {torch.mean(spks_r)*100:.2f}%")
    
    plt.hist(curv_l.flatten(), bins=100, alpha=0.5, label='Left', color='blue', range=(0, 2))
    plt.hist(curv_r.flatten(), bins=100, alpha=0.5, label='Right', color='red', range=(0, 2))
    plt.title(r"Curvature ($\kappa$) Distribution Comparison")
    plt.legend(); plt.show()


def run_training_pipeline():
    print("\n--- Phase 4: SNN Model Training Pipeline ---")
    
    # 1. 데이터 로드 및 전처리 (Subject 1)
    loader = EEGLoader()
    raw = loader.fetch_and_load(subject=1)
    analytic_data, _ = loader.process_to_analytic(raw)
    
    # 2. 기하학적 특징 및 스파이크 추출
    curvature = GeometryExtractor.calculate_curvature(analytic_data)
    encoder = GeometricSpikeEncoder(threshold_type='std')
    spikes, _ = encoder.threshold_encoding(curvature) # (Channels, Time)
    
    # 3. 라벨링 및 데이터셋 생성
    # EEGBCI의 이벤트를 가져와서 스파이크 시간대와 매칭
    events, event_id = mne.events_from_annotations(raw)
    dataset = GeometricEEGDataset(spikes, events, event_id, window_size=160) # 1초 윈도우
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 4. 모델 설정 (채널 수 = 입력 뉴런 수)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometricSNN(num_inputs=spikes.shape[0], num_hidden=64, num_outputs=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Training Loop
    print(f"Starting training on {device}...")
    model.train()
    for epoch in range(10): # 예시로 10 에포크
        total_loss = 0
        for data, target in train_loader:
            # data: (Batch, Time, Channels) -> (Time, Batch, Channels) 변환 필요
            data = data.transpose(0, 1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            spk_out, _ = model(data) # (Time, Batch, Output)
            
            # 마지막 시점의 출력을 사용하거나 전체 시간 평균(Rate) 사용
            output = torch.mean(spk_out, dim=0) 
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

    print("--- Training Execution Finished ---")
    
    
if __name__ == "__main__":
    # 원하는 함수만 주석 해제해서 실행해!
    # run_research_pipeline()
    # run_comparison()
    # run_quantitative_comparison()
    run_training_pipeline()
    '''
    --- Encoding with threshold: 0.5199 ---
    --- Encoding with threshold: 0.5199 ---

    [Result] Spiking Density
    - Left Hand : 18.86%
    - Right Hand: 18.99%
    - Difference: 0.13%
    '''