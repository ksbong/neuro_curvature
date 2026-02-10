import numpy as np

class RealisticNoiseInjector:
    def __init__(self, srate=160):
        self.srate = srate

    def add_white_noise(self, X, strength=1.0):
        """기본 가우시안 노이즈"""
        noise = np.random.randn(*X.shape)
        return X + strength * noise

    def add_eog_drift(self, X, strength=1.0):
        """눈 깜빡임 (저주파수 큰 진폭)"""
        # EOG는 보통 0.5~4Hz 대역의 큰 움직임
        t = np.arange(X.shape[-1]) / self.srate
        n_epochs, n_ch, n_times = X.shape
        
        # 채널마다 위상을 다르게 하여 랜덤성 부여
        drift = np.zeros_like(X)
        for i in range(n_epochs):
            for c in range(n_ch):
                freq = np.random.uniform(0.5, 3.0) # 랜덤 저주파
                phase = np.random.uniform(0, 2*np.pi)
                # EOG는 앞쪽 채널에 강하지만, 여기서는 단순화를 위해 전체 적용하되 랜덤 스케일
                scale = np.random.uniform(0.5, 1.5)
                drift[i, c] = np.sin(2 * np.pi * freq * t + phase) * scale
        
        return X + (strength * 5.0) * drift # EOG는 원래 신호보다 훨씬 큼

    def add_line_noise(self, X, freq=60.0, strength=0.5):
        t = np.arange(X.shape[-1]) / self.srate
        noise = np.sin(2 * np.pi * freq * t)
        return X + strength * noise

    def add_emg_burst(self, X, strength=1.0, prob=0.1):
        noise = np.random.randn(*X.shape)
        mask = np.random.rand(*X.shape) < prob
        return X + (noise * mask * strength)

    def get_noisy_data(self, X, noise_type='realistic', level=0.0):
        """
        noise_type: 'gaussian', 'eog', 'realistic'
        level: 0.0 ~ 2.0 (강도)
        """
        if level <= 0: return X.copy()
        
        X_n = X.copy()
        
        if noise_type == 'gaussian':
            # 단순 백색 잡음
            X_n = self.add_white_noise(X_n, strength=level)
            
        elif noise_type == 'eog':
            # 눈 깜빡임 집중 테스트 (CSP가 보통 여기서 망가짐)
            X_n = self.add_eog_drift(X_n, strength=level)
            
        elif noise_type == 'realistic':
            # 종합 상황
            X_n = self.add_line_noise(X_n, strength=0.2 * level)
            X_n = self.add_eog_drift(X_n, strength=1.0 * level)
            X_n = self.add_emg_burst(X_n, strength=0.5 * level)
            X_n = self.add_white_noise(X_n, strength=0.1 * level)
            
        return X_n