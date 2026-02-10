import numpy as np
from scipy.signal import hilbert, savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

class GeometryFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=25, polyorder=3):
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
            
            # [수정됨] 채널을 합치지 않고, 각 채널별로 루프를 돕니다.
            for ch in range(n_channels):
                # 채널별 신호 추출 (1D signal)
                x_ch = X[i, ch, :]
                
                # 1. 힐베르트 변환 (Analytic Signal)
                z = hilbert(x_ch)
                
                # 2. Savitzky-Golay 미분
                dz = savgol_filter(z, self.window_length, self.polyorder, deriv=1)
                ddz = savgol_filter(z, self.window_length, self.polyorder, deriv=2)
                
                # 3. 스칼라 값으로 변환 (Vector -> Scalar)
                # 1D 신호이므로 axis=0 등의 합산이 필요 없음 (단일 값)
                mag_dz_sq = np.abs(dz)**2 + 1e-9
                mag_ddz_sq = np.abs(ddz)**2
                dot_dz_ddz = np.abs(dz * np.conj(ddz))**2
                
                # 4. 복소 곡률 (Complex Curvature) - 시간 축에 대한 배열
                kappa = np.sqrt(np.maximum(0, mag_dz_sq * mag_ddz_sq - dot_dz_ddz)) / (mag_dz_sq**1.5)
                
                # 5. Tangling (꼬임 지표)
                step = 4
                z_sub = z[::step]
                dz_sub = dz[::step]
                
                # 거리 행렬 계산 (Broadcasting)
                dist_z = np.abs(z_sub[:, None] - z_sub[None, :])**2
                dist_dz = np.abs(dz_sub[:, None] - dz_sub[None, :])**2
                
                Q_matrix = dist_dz / (dist_z + 1e-6)
                np.fill_diagonal(Q_matrix, 0) # 자기 자신과의 거리는 제외
                Q_t = np.max(Q_matrix, axis=1) # 각 시점별 최대 꼬임
                
                # 6. 특징 추출 (채널별 통계량)
                # 이 채널의: [곡률 중간값, 평균 에너지, 최대 꼬임, 평균 꼬임, 로그 분산]
                ch_feats = [
                    np.median(kappa), 
                    np.mean(np.abs(dz)), 
                    np.max(Q_t), 
                    np.mean(Q_t),
                    np.log(np.var(x_ch) + 1e-9)
                ]
                epoch_features.extend(ch_feats)
            
            # 모든 채널의 특징을 일렬로 연결 (Concatenate)
            # 결과 벡터 길이 = n_channels * 5
            features.append(epoch_features)
            
        return np.array(features)