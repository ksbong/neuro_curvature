import numpy as np

class RealisticNoiseInjector:
    """
    [Artifact Injector]
    단순 White Noise뿐만 아니라, BCI를 파괴하는 실제 아티팩트(EOG, EMG)를 생성합니다.
    """
    def get_noisy_data(self, X, noise_type='eog', level=0.0):
        if level <= 0:
            return X
        
        n_epochs, n_channels, n_times = X.shape
        X_noisy = X.copy()
        
        # 기준 신호 크기 (Standard Deviation)
        signal_std = np.std(X)
        
        if noise_type == 'white':
            # 기존: 백색 소음 (전체적인 SNR 저하)
            noise = np.random.normal(0, signal_std * level, X.shape)
            X_noisy += noise
            
        elif noise_type == 'eog':
            # [핵심] EOG (눈 깜빡임): 저주파(1~3Hz) 대용량 드리프트
            # 주로 전두엽(Frontal) 쪽에 끼지만, 여기선 랜덤 채널에 섞음
            # level 1.0 = 신호 크기의 5배 (눈 깜빡임은 뇌파보다 훨씬 큼)
            amplitude = signal_std * level * 5 
            
            t = np.linspace(0, 2*np.pi, n_times)
            for i in range(n_epochs):
                # 에포크마다 1~2개의 랜덤 채널에 눈 깜빡임 발생
                blink_channels = np.random.choice(n_channels, size=2, replace=False)
                for ch in blink_channels:
                    # 느린 사인파 + 약간의 랜덤성
                    blink = np.sin(t * np.random.uniform(1, 3)) 
                    # 반파장만 사용하여 '깜빡'하는 모양 흉내
                    blink = np.abs(blink) * amplitude
                    X_noisy[i, ch, :] += blink

        elif noise_type == 'emg':
            # EMG (근육): 고주파(30Hz+) 버스트
            amplitude = signal_std * level * 3
            for i in range(n_epochs):
                muscle_channels = np.random.choice(n_channels, size=3, replace=False)
                for ch in muscle_channels:
                    # 랜덤한 시점에 '팍' 하고 튀는 노이즈
                    burst_start = np.random.randint(0, n_times - 50)
                    burst = np.random.normal(0, amplitude, 50) # 50ms 버스트
                    X_noisy[i, ch, burst_start:burst_start+50] += burst
                    
        return X_noisy