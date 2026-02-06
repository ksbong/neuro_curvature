import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import hilbert, savgol_filter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. 기하학 특징 추출 클래스 (Savitzky-Golay 적용) ---
class GeometryFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=25, polyorder=3):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        n_epochs = X.shape[0]
        
        for i in range(n_epochs):
            # A. 힐베르트 변환 (Analytic Signal)
            z = hilbert(X[i], axis=-1)
            
            # B. Savitzky-Golay 스무딩 미분 (노이즈 억제)
            dz = savgol_filter(z, self.window_length, self.polyorder, deriv=1, axis=-1)
            ddz = savgol_filter(z, self.window_length, self.polyorder, deriv=2, axis=-1)
            
            # C. 복소 곡률 (Curvature)
            mag_dz_sq = np.sum(np.abs(dz)**2, axis=0) + 1e-9
            mag_ddz_sq = np.sum(np.abs(ddz)**2, axis=0)
            dot_dz_ddz = np.abs(np.sum(dz * np.conj(ddz), axis=0))**2
            
            kappa = np.sqrt(np.maximum(0, mag_dz_sq * mag_ddz_sq - dot_dz_ddz)) / (mag_dz_sq**1.5)
            
            # D. Tangling Index (다운샘플링)
            step = 4
            z_sub = z[:, ::step]
            dz_sub = dz[:, ::step]
            
            dist_z = np.sum(np.abs(z_sub[:, :, None] - z_sub[:, None, :])**2, axis=0)
            dist_dz = np.sum(np.abs(dz_sub[:, :, None] - dz_sub[:, None, :])**2, axis=0)
            
            Q_matrix = dist_dz / (dist_z + 1e-6)
            Q_t = np.max(Q_matrix, axis=1)
            
            # E. 특징 벡터 구성
            f = [
                np.median(kappa),       # 곡률 중간값
                np.mean(np.abs(dz)),    # 평균 에너지
                np.max(Q_t),            # 최대 꼬임
                np.mean(Q_t)            # 평균 꼬임
            ]
            f.extend(np.log(np.var(X[i], axis=1) + 1e-9)) # 채널별 분산
            
            features.append(f)
            
        return np.array(features)

# --- 2. 실험 설정 (PhysioNet Dataset) ---
subjects = [1, 2, 3, 4, 5]
noise_levels = [0.0, 0.4, 0.8, 1.2, 1.6]

results_all = {
    'Geometry Only': {nl: [] for nl in noise_levels},
    'Standard CSP': {nl: [] for nl in noise_levels},
    'Hybrid (CSP+Geom)': {nl: [] for nl in noise_levels}
}

print(f"=== PhysioNet Public Benchmark Test (N={len(subjects)}) ===")
print("System: Hybrid Architecture (Energy + Topology)")

for sub in subjects:
    print(f"\n[Subject {sub}] Processing...")
    try:
        # PhysioNet MI Data Load
        runs = [4, 8, 12] # Motor Imagery: Left vs Right Fist
        raw_fnames = eegbci.load_data(sub, runs, verbose=False)
        raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
        for f in raw_fnames[1:]:
            raw.append(mne.io.read_raw_edf(f, preload=True, verbose=False))
        
        eegbci.standardize(raw)
        raw.filter(8., 30., fir_design='firwin', verbose=False)
        picks = mne.pick_channels(raw.info['ch_names'], ['C3', 'Cz', 'C4', 'FC3', 'FC4', 'CP3', 'CP4'])
        
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)
        epochs = mne.Epochs(raw, events, event_id={'left': 2, 'right': 3}, 
                            tmin=0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose=False)
        
        X_orig, y = epochs.get_data(), epochs.events[:, -1]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for nl in noise_levels:
            noise = np.random.randn(*X_orig.shape)
            X_noisy = X_orig + nl * noise
            
            # 1. Geometry Only
            geom_pipe = Pipeline([
                ('geom', GeometryFeatureExtractor(window_length=25, polyorder=3)),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=10))
            ])
            score_g = cross_val_score(geom_pipe, X_noisy, y, cv=cv).mean()
            results_all['Geometry Only'][nl].append(score_g)
            
            # 2. Standard CSP
            csp_pipe = Pipeline([
                ('csp', CSP(n_components=4, log=True, norm_trace=False)),
                ('svm', SVC(kernel='linear'))
            ])
            score_c = cross_val_score(csp_pipe, X_noisy, y, cv=cv).mean()
            results_all['Standard CSP'][nl].append(score_c)
            
            # 3. Hybrid (CSP + Geometry)
            hybrid_feats = FeatureUnion([
                ('csp', CSP(n_components=4, log=True, norm_trace=False)),
                ('geom', GeometryFeatureExtractor(window_length=25, polyorder=3))
            ])
            hybrid_pipe = Pipeline([
                ('union', hybrid_feats),
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=10))
            ])
            score_h = cross_val_score(hybrid_pipe, X_noisy, y, cv=cv).mean()
            results_all['Hybrid (CSP+Geom)'][nl].append(score_h)
            
            print(f"  Noise {nl:.1f}σ -> Hybrid: {score_h*100:.1f}% (CSP: {score_c*100:.1f}%, Geom: {score_g*100:.1f}%)")
            
    except Exception as e:
        print(f"Subject {sub} Error: {e}")

# --- 3. 최종 결과 시각화 (논문 스타일) ---
print("\n=== 결과 집계 및 그래프 생성 ===")

avg_results = {k: [] for k in results_all.keys()}
for nl in noise_levels:
    for method in avg_results.keys():
        # 단위를 %로 변환 ( * 100 )
        avg_results[method].append(np.mean(results_all[method][nl]) * 100)

plt.figure(figsize=(10, 6))
markers = {'Geometry Only': 'o-', 'Standard CSP': 's--', 'Hybrid (CSP+Geom)': 'D-.'}
colors = {'Geometry Only': '#1f77b4', 'Standard CSP': '#ff7f0e', 'Hybrid (CSP+Geom)': '#2ca02c'}

for method, scores in avg_results.items():
    plt.plot(noise_levels, scores, markers[method], color=colors[method], label=method, linewidth=2.5, markersize=8)

# Chance Level Line
plt.axhline(50, color='red', linestyle=':', linewidth=2, label='Chance Level (50%)')

plt.title(f'Robustness Analysis: Hybrid Architecture (N={len(subjects)})', fontsize=16, fontweight='bold')
plt.xlabel('Gaussian Noise Intensity ($\sigma$)', fontsize=13) # 시그마 단위 명시
plt.ylabel('Mean Decoding Accuracy (%)', fontsize=13) # 퍼센트 단위 명시
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.yticks(np.arange(30, 90, 5)) # Y축 눈금 디테일
plt.tight_layout()
plt.show()

print("\n[완료] 그래프의 Y축은 이제 '정확도(%)'를, X축은 '노이즈 강도(σ)'를 나타냅니다.")