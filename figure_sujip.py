import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import hilbert, savgol_filter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from mne.time_frequency import tfr_multitaper

# --- 스타일 설정 ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. 기하학 특징 추출 클래스 (핵심 알고리즘)
# ==========================================
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
            z = hilbert(X[i], axis=-1)
            # Savitzky-Golay로 미분 (노이즈 제거)
            dz = savgol_filter(z, self.window_length, self.polyorder, deriv=1, axis=-1)
            ddz = savgol_filter(z, self.window_length, self.polyorder, deriv=2, axis=-1)
            
            mag_dz_sq = np.sum(np.abs(dz)**2, axis=0) + 1e-9
            mag_ddz_sq = np.sum(np.abs(ddz)**2, axis=0)
            dot_dz_ddz = np.abs(np.sum(dz * np.conj(ddz), axis=0))**2
            
            # 복소 곡률 (Complex Curvature)
            kappa = np.sqrt(np.maximum(0, mag_dz_sq * mag_ddz_sq - dot_dz_ddz)) / (mag_dz_sq**1.5)
            
            # Tangling (꼬임 지표)
            step = 4
            z_sub = z[:, ::step]
            dz_sub = dz[:, ::step]
            dist_z = np.sum(np.abs(z_sub[:, :, None] - z_sub[:, None, :])**2, axis=0)
            dist_dz = np.sum(np.abs(dz_sub[:, :, None] - dz_sub[:, None, :])**2, axis=0)
            Q_matrix = dist_dz / (dist_z + 1e-6)
            Q_t = np.max(Q_matrix, axis=1)
            
            f = [np.median(kappa), np.mean(np.abs(dz)), np.max(Q_t), np.mean(Q_t)]
            f.extend(np.log(np.var(X[i], axis=1) + 1e-9))
            features.append(f)
        return np.array(features)

# ==========================================
# 2. 데이터 처리 및 캡처 (통합 로직 적용)
# ==========================================
subjects = [1, 2, 3, 4, 5]
noise_levels = [0.0, 0.4, 0.8, 1.2, 1.6]
robustness_data = []

# 시각화용 데이터 저장소 (Subject 1만 저장)
viz_data = {'manifold_X': None, 'manifold_y': None, 'features_df': None}
viz_epochs = None 

# [NEW] 모든 피험자의 예측 결과 모으기용 리스트
all_y_true = []
all_y_pred = []

print(f"=== Processing Logic (N={len(subjects)}) ===")

for sub in subjects:
    print(f"\n[Subject {sub}] Processing...")
    
    try:
        # --- 공통 데이터 로드 로직 ---
        runs = [4, 8, 12] 
        raw_fnames = eegbci.load_data(sub, runs, verbose=False)
        raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
        for f in raw_fnames[1:]:
            raw.append(mne.io.read_raw_edf(f, preload=True, verbose=False))
        
        # 전처리 및 에포크 생성
        eegbci.standardize(raw)  
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        raw.filter(8., 30., fir_design='firwin', verbose=False)
        picks = mne.pick_channels(raw.info['ch_names'], ['C3', 'Cz', 'C4', 'FC3', 'FC4', 'CP3', 'CP4'])
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)
        epochs = mne.Epochs(raw, events, event_id={'left': 2, 'right': 3}, 
                            tmin=0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose=False)
        
        X_orig, y = epochs.get_data(), epochs.events[:, -1]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- Subject 1일 경우: 논문용 시각화 데이터 따로 캡처 ---
        if sub == 1:
            print(" >> Capturing Subject 1 Data for Visualization...")
            viz_epochs = epochs.copy()
            
            # Manifold용 PCA
            z_clean = hilbert(X_orig, axis=-1)
            n_epochs_viz, n_ch_viz, n_times_viz = z_clean.shape
            X_flat = np.real(z_clean).transpose(0, 2, 1).reshape(-1, n_ch_viz)
            pca_vis = PCA(n_components=3)
            X_pca_flat = pca_vis.fit_transform(X_flat)
            viz_data['manifold_X'] = X_pca_flat.reshape(n_epochs_viz, n_times_viz, 3)
            viz_data['manifold_y'] = y
            
            # Feature Distribution용 DF
            extractor = GeometryFeatureExtractor()
            feats_real = extractor.transform(X_orig)
            viz_data['features_df'] = pd.DataFrame({
                'Complex Curvature ($\kappa$)': feats_real[:, 0],
                'Tangling ($Q$)': feats_real[:, 2],
                'Condition': ['Left Hand' if label==2 else 'Right Hand' for label in y]
            })

        # -----------------------------------------------------------
        # [NEW] Confusion Matrix용 데이터 누적 (Noise 없는 깨끗한 데이터 기준)
        # -----------------------------------------------------------
        # 모델 파이프라인 정의 (Hybrid)
        hybrid_pipe_cm = Pipeline([
            ('union', FeatureUnion([('csp', CSP(n_components=4, log=True)), ('geom', GeometryFeatureExtractor())])), 
            ('s', StandardScaler()), ('svm', SVC(C=10))
        ])
        
        # 현재 피험자에 대해 CV 예측 수행 (Train/Test 쪼개지 않고 전체 데이터에 대한 OOF 예측값 획득)
        y_pred_sub = cross_val_predict(hybrid_pipe_cm, X_orig, y, cv=cv)
        
        # 리스트에 저장
        all_y_true.extend(y)
        all_y_pred.extend(y_pred_sub)
        # -----------------------------------------------------------

        # --- 노이즈 강건성 테스트 (기존 유지) ---
        for nl in noise_levels:
            noise = np.random.randn(*X_orig.shape)
            X_noisy = X_orig + nl * noise
            
            # 파이프라인들 정의
            hybrid_pipe = Pipeline([
                ('union', FeatureUnion([('csp', CSP(n_components=4, log=True)), ('geom', GeometryFeatureExtractor())])), 
                ('s', StandardScaler()), ('svm', SVC(C=10))
            ])
            csp_pipe = Pipeline([('csp', CSP(n_components=4, log=True, norm_trace=False)), ('svm', SVC(kernel='linear'))])
            geom_pipe = Pipeline([('geom', GeometryFeatureExtractor()), ('s', StandardScaler()), ('svm', SVC(C=10))])

            s_h = cross_val_score(hybrid_pipe, X_noisy, y, cv=cv).mean()
            s_c = cross_val_score(csp_pipe, X_noisy, y, cv=cv).mean()
            s_g = cross_val_score(geom_pipe, X_noisy, y, cv=cv).mean()
            
            robustness_data.append({'Noise': nl, 'Accuracy': s_g*100, 'Method': 'Geometry Only'})
            robustness_data.append({'Noise': nl, 'Accuracy': s_c*100, 'Method': 'Standard CSP'})
            robustness_data.append({'Noise': nl, 'Accuracy': s_h*100, 'Method': 'Hybrid (Ours)'})

    except Exception as e:
        print(f"Error processing Subject {sub}: {e}")
        continue

df_results = pd.DataFrame(robustness_data)

# ==========================================
# 3. 최종 논문용 Figure 생성
# ==========================================
print("\n=== Generating Figures ===")

# Check if data capture was successful
if viz_data['manifold_y'] is None:
    raise ValueError("CRITICAL: Subject 1 Data was not captured. Cannot generate figures.")

# --- Figure 1: 3D Manifold ---
print("1. Drawing 3D Manifold...")
fig_3d = go.Figure()
colors = {2: '#1f77b4', 3: '#d62728'}
names = {2: 'Left Hand', 3: 'Right Hand'}

y_labels_safe = np.atleast_1d(viz_data['manifold_y'])
indices_left = np.where(y_labels_safe == 2)[0]
indices_right = np.where(y_labels_safe == 3)[0]

selected_indices = []
if len(indices_left) > 0: selected_indices.append(indices_left[0])
if len(indices_right) > 0: selected_indices.append(indices_right[0])

for idx in selected_indices:
    label = viz_data['manifold_y'][idx]
    traj = viz_data['manifold_X'][idx]
    
    fig_3d.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines',
        name=names[label], line=dict(color=colors[label], width=5), opacity=0.9
    ))

fig_3d.update_layout(
    title="<b>Fig 1. Neural Manifold Trajectories</b>", 
    width=800, height=600,
    scene=dict(xaxis_title='PC 1 (a.u.)', yaxis_title='PC 2 (a.u.)', zaxis_title='PC 3 (a.u.)')
)
fig_3d.show()

# --- Figure 2: Feature Distribution ---
print("2. Drawing Feature Distribution...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.violinplot(data=viz_data['features_df'], x='Condition', y='Complex Curvature ($\kappa$)', palette="muted")
plt.ylabel('Curvature $\kappa$ (a.u.)')
plt.subplot(1, 2, 2)
sns.violinplot(data=viz_data['features_df'], x='Condition', y='Tangling ($Q$)', palette="muted")
plt.ylabel('Tangling Index $Q$ (a.u.)')
plt.tight_layout()
plt.show()

# --- Figure 3: Robustness ---
print("3. Drawing Robustness Curve...")
if not df_results.empty:
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_results, x='Noise', y='Accuracy', hue='Method', style='Method',
                 palette={'Hybrid (Ours)': 'green', 'Standard CSP': 'orange', 'Geometry Only': 'blue'}, 
                 markers=True, dashes=False, linewidth=2.5, errorbar='sd') # ci 대신 errorbar 권장 (최신 seaborn)
    plt.axhline(50, color='gray', linestyle=':', label='Chance Level')
    plt.xlabel('Noise Intensity ($\sigma$)')
    plt.ylabel('Decoding Accuracy (%)')
    plt.ylim(40, 100)
    plt.legend()
    plt.show()

# --- Figure 4: Topomaps ---
print("4. Drawing Topomaps...")
if viz_epochs is not None:
    csp_vis = CSP(n_components=4, log=True, norm_trace=False)
    X_clean = viz_epochs.get_data()
    csp_vis.fit(X_clean, viz_epochs.events[:, -1])
    try:
        csp_vis.plot_patterns(viz_epochs.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5, show=True)
    except Exception as e:
        print(f"Topomap plot skipped: {e}")

# --- Figure 5: Spectrogram ---
print("5. Drawing Spectrogram...")
if viz_epochs is not None:
    freqs = np.arange(8, 30, 1)
    n_cycles = freqs / 2.
    power = tfr_multitaper(viz_epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    power.plot([viz_epochs.ch_names.index('C3')], baseline=None, mode='logratio', 
               axes=ax[0], show=False, colorbar=True, dB=True)
    ax[0].set_title('Channel C3 (Left Motor Cortex)')
    
    power.plot([viz_epochs.ch_names.index('C4')], baseline=None, mode='logratio', 
               axes=ax[1], show=False, colorbar=True, dB=True)
    ax[1].set_title('Channel C4 (Right Motor Cortex)')
    plt.tight_layout()
    plt.show()

# --- Figure 6: Confusion Matrix (All Subjects) ---
print("6. Drawing Confusion Matrix (All Subjects Aggregated)...")
if all_y_true:
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Sample Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (All Subjects, N={len(subjects)})\nTotal Samples: {len(y_true_all)}')
    plt.show()
else:
    print("No data collected for Confusion Matrix.")