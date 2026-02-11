import numpy as np
from scipy.signal import hilbert, savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

class GeometryFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=25, polyorder=3):
        # 님 원래 설정 (window=25, poly=3) 그대로 복구
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = X.shape
        features = []
        
        for i in range(n_epochs):
            epoch_features = []
            # [핵심 수정] 채널을 합치지(sum) 않고 반복문으로 각각 추출
            for ch in range(n_channels):
                x_ch = X[i, ch, :] # (n_times,) 1D array
                
                # --- 님의 Original Logic 시작 ---
                # 1. Analytic Signal
                z = hilbert(x_ch) 
                
                # 2. Savitzky-Golay (Noise reduction)
                dz = savgol_filter(z, self.window_length, self.polyorder, deriv=1)
                ddz = savgol_filter(z, self.window_length, self.polyorder, deriv=2)
                
                # 3. Complex Curvature (Original Formula)
                # 1D 배열이므로 axis=0 합산 없이 바로 계산 (점 단위 연산)
                mag_dz_sq = np.abs(dz)**2 + 1e-9
                mag_ddz_sq = np.abs(ddz)**2
                dot_dz_ddz = np.abs(dz * np.conj(ddz))**2
                
                # 분모/분자 계산
                numerator = np.sqrt(np.maximum(0, mag_dz_sq * mag_ddz_sq - dot_dz_ddz))
                denominator = mag_dz_sq**1.5
                
                kappa = numerator / denominator
                
                # 4. Tangling (Original Logic)
                step = 4
                z_sub = z[::step]
                dz_sub = dz[::step]
                
                # 1D 거리 계산 (Broadcasting)
                dist_z = np.abs(z_sub[:, None] - z_sub[None, :])**2
                dist_dz = np.abs(dz_sub[:, None] - dz_sub[None, :])**2
                
                Q_matrix = dist_dz / (dist_z + 1e-6)
                np.fill_diagonal(Q_matrix, 0)
                Q_t = np.max(Q_matrix, axis=1)
                
                # 5. Feature Vector (채널별로 저장)
                # 로그 변환(np.log)은 데이터 분포상 SVM을 위해 유지하는 게 좋습니다.
                f = [
                    np.median(kappa),       # 곡률 중간값
                    np.mean(np.abs(dz)),    # 에너지 (Phase Velocity)
                    np.max(Q_t),            # 최대 꼬임
                    np.mean(Q_t),           # 평균 꼬임
                    np.log(np.var(x_ch) + 1e-9) # Log Variance
                ]
                epoch_features.extend(f)
                # --- 님의 Original Logic 끝 ---

            # 결과: (Epochs, Channels * 5) 형태의 벡터
            features.append(epoch_features)
            
        return np.array(features)