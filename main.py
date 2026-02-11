import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

# 모듈 임포트
from src.features import GeometryFeatureExtractor 
from src.noise import RealisticNoiseInjector
from src.data_loader import load_multiclass_data

# --- 설정 ---
subjects = [1] 
# 노이즈 레벨 복구
noise_levels = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0] 
scenarios = ['realistic'] 

results = []

print("=== NeuroCurvature Original Method (Channel-wise) ===")

for sub in subjects:
    print(f"\n[Subject {sub}] Loading Data...")
    X, y = load_multiclass_data(sub) # X shape: (Epochs, 7, Times)
    injector = RealisticNoiseInjector()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for scene in scenarios:
        for nl in noise_levels:
            X_curr = injector.get_noisy_data(X, noise_type=scene, level=nl)

            models = {
                # 비교군 CSP
                'CSP': Pipeline([
                    ('csp', CSP(n_components=6, log=True, norm_trace=False)), 
                    ('svm', SVC(kernel='linear'))
                ]),
                # 님의 Original Method (채널별 특징 -> SVM)
                'Geometry (Ours)': Pipeline([
                    ('geom', GeometryFeatureExtractor()), 
                    ('s', StandardScaler()), 
                    ('svm', SVC(C=10, kernel='rbf')) # RBF 커널 유지
                ]),
                # Hybrid
                'Hybrid': Pipeline([
                    ('union', FeatureUnion([
                        ('csp', CSP(n_components=6, log=True)), 
                        ('geom', GeometryFeatureExtractor())
                    ])), 
                    ('s', StandardScaler()), 
                    ('svm', SVC(C=10))
                ])
            }

            for name, model in models.items():
                start_time = time.time()
                cv_results = cross_validate(model, X_curr, y, cv=cv, return_train_score=False)
                elapsed = time.time() - start_time
                avg_acc = cv_results['test_score'].mean()
                
                results.append({
                    'Noise Level': nl, 
                    'Method': name, 
                    'Accuracy (%)': avg_acc * 100
                })
            
            print(f"Noise {nl} Done.")

# --- 시각화 ---
df = pd.DataFrame(results)
plt.figure(figsize=(8, 6))
sns.lineplot(data=df, x='Noise Level', y='Accuracy (%)', hue='Method', style='Method', markers=True, linewidth=2.5)
plt.axhline(33.3, color='gray', linestyle='--')
plt.title(f'Robustness Analysis (3-Class, Subject {subjects[0]})')
plt.show()