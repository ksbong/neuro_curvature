from src.core.preprocess import EEGLoader
from src.core.geometry import GeometryExtractor
import matplotlib.pyplot as plt
import numpy as np

def plot_complex_trajectory_3d(z, channel_idx=0, duration_sec=1.0, sfreq=160.0):
    """
    단일 채널의 복소 궤적을 3차원(시간, 실수, 허수)으로 시각화.
    * 단위 표시 추가됨
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 샘플 수 계산
    n_samples = int(duration_sec * sfreq)
    z_slice = z[channel_idx, :n_samples]
    
    real = z_slice.real
    imag = z_slice.imag
    time = np.arange(len(z_slice)) / sfreq  # 시간축 (초 단위)
    
    # 3D 선 그래프
    ax.plot(real, imag, time, label=f'Ch {channel_idx}', alpha=0.8, linewidth=1.5)
    
    # [중요] 단위(Unit) 명시
    ax.set_xlabel(r'Real Amplitude [$\mu V$]')
    ax.set_ylabel(r'Imaginary Amplitude [$\mu V$]')
    ax.set_zlabel('Time [s]')
    ax.set_title(f'3D Phase Space Trajectory (Ch {channel_idx})')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_3d_trajectories(z_list, labels, channel_idx=0, duration_sec=1.0, sfreq=160.0, overlay=False):
    """
    여러 신호의 궤적을 비교.
    overlay=True: 한 그래프에 겹쳐서 그림 (위상 차이 확인용)
    overlay=False: 같은 스케일의 서브플롯으로 나란히 그림 (형태 비교용)
    """
    n_samples = int(duration_sec * sfreq)
    time = np.arange(n_samples) / sfreq
    
    if overlay:
        # 겹쳐 그리기 모드
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for z, label in zip(z_list, labels):
            z_slice = z[channel_idx, :n_samples]
            ax.plot(z_slice.real, z_slice.imag, time, label=label, alpha=0.7)
            
        ax.set_xlabel(r'Real [$\mu V$]')
        ax.set_ylabel(r'Imaginary [$\mu V$]')
        ax.set_zlabel('Time [s]')
        ax.set_title(f'Trajectory Comparison (Overlay) - Ch {channel_idx}')
        ax.legend()
        plt.show()
        
    else:
        # 나란히 그리기 모드 (Scale 공유)
        fig = plt.figure(figsize=(6 * len(z_list), 6))
        
        # 축 스케일 통일을 위한 Min/Max 계산
        all_real = np.concatenate([z[channel_idx, :n_samples].real for z in z_list])
        all_imag = np.concatenate([z[channel_idx, :n_samples].imag for z in z_list])
        r_min, r_max = all_real.min(), all_real.max()
        i_min, i_max = all_imag.min(), all_imag.max()
        
        for i, (z, label) in enumerate(zip(z_list, labels)):
            ax = fig.add_subplot(1, len(z_list), i+1, projection='3d')
            z_slice = z[channel_idx, :n_samples]
            
            ax.plot(z_slice.real, z_slice.imag, time, alpha=0.8)
            
            # 스케일 고정
            ax.set_xlim(r_min, r_max)
            ax.set_ylim(i_min, i_max)
            ax.set_zlim(0, duration_sec)
            
            ax.set_xlabel(r'Real [$\mu V$]')
            ax.set_ylabel(r'Imaginary [$\mu V$]')
            ax.set_zlabel('Time [s]')
            ax.set_title(f'{label}')
            
        plt.suptitle(f'Trajectory Comparison (Side-by-Side) - Ch {channel_idx}')
        plt.tight_layout()
        plt.show()
        
def run_research_pipeline():
    """
    [연구용 파이프라인]
    데이터 로드 -> 힐베르트 변환 -> 기하학적 특징 추출(곡률, 비틀림, 속력) -> 3D 시각화
    """
    print("\n--- 🧪 Phase 1: Geometric Analysis Pipeline ---")
    
    # 1. 데이터 로드 및 전처리
    loader = EEGLoader()
    raw = loader.fetch_and_load(subjects=[1]) # 피험자 1번 데이터
    
    # 2. 힐베르트 변환 (Alpha~Beta 대역 집중: 8~30Hz 예시)
    # *광대역 신호가 기하학적 특성이 더 잘 보일 수도 있으니 필터 범위 조절 가능
    z, info = loader.process_to_analytic(raw, l_freq=8.0, h_freq=30.0)
    sfreq = info['sfreq']
    
    # 3. 새로운 기하학적 특징 추출 (New Features!)
    print("Computing Geometric Features...")
    curvature = GeometryExtractor.calculate_curvature(z)
    velocity = GeometryExtractor.calculate_complex_velocity(z, sfreq=sfreq)
    torsion = GeometryExtractor.calculate_torsion_3d(z, sfreq=sfreq)
    
    # 4. 시각화 1: 3D 위상 공간 궤적 (Time-Real-Imag)
    target_ch = 10 # 시각화할 채널 인덱스
    print(f"Visualizing 3D Trajectory for Channel {target_ch}...")
    plot_complex_trajectory_3d(z, channel_idx=target_ch, duration_sec=2.0, sfreq=sfreq)
    
    # 5. 시각화 2: Feature 비교 (원본 vs 곡률 vs 비틀림)
    print("Comparing Features...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    subset = int(sfreq * 2.0) # 2초 구간
    t_axis = np.arange(subset) / sfreq
    
    # (1) 원본 진폭
    axes[0].plot(t_axis, np.abs(z[target_ch, :subset]), color='k')
    axes[0].set_title("Instantaneous Amplitude (Envelope)")
    axes[0].set_ylabel(r"Amp [$\mu V$]")
    
    # (2) 속력 (Velocity)
    axes[1].plot(t_axis, velocity[target_ch, :subset], color='orange')
    axes[1].set_title("Complex Velocity (Speed in Phase Space)")
    axes[1].set_ylabel("Speed")
    
    # (3) 곡률 (Curvature)
    axes[2].plot(t_axis, curvature[target_ch, :subset], color='blue')
    axes[2].set_title("Curvature (2D Plane Bending)")
    axes[2].set_ylabel(r"$\kappa$")
    
    # (4) 비틀림 (Torsion) - 3D 특성
    axes[3].plot(t_axis, torsion[target_ch, :subset], color='red')
    axes[3].set_title("Torsion (3D Twisting)")
    axes[3].set_ylabel(r"$\tau$")
    axes[3].set_xlabel("Time [s]")
    
    plt.tight_layout()
    plt.show()