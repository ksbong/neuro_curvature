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