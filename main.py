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

# --- [설정] ---
subjects = [1]  # 테스트용 (시간 많으면 [1,2,3,4,5] 권장)
noise_levels = np.linspace(0, 2.0, 11) # 0.0 ~ 2.0 까지 10등분 (촘촘하게!)
scenarios = ['gaussian', 'eog', 'realistic'] # 3가지 테스트 시나리오

results = []

print(f"=== NeuroCurvature Comprehensive Benchmark ===")
print(f"Scenarios: {scenarios}")
print(f"Noise Levels: {noise_levels}")

for sub in subjects:
    print(f"\n[Subject {sub}] Loading Data...")
    X, y = load_multiclass_data(sub)
    injector = RealisticNoiseInjector()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 시나리오별 루프
    for scene in scenarios:
        print(f"  >> Running Scenario: {scene.upper()}")
        
        for nl in noise_levels:
            # 노이즈 생성
            X_curr = injector.get_noisy_data(X, noise_type=scene, level=nl)

            # 모델 정의
            models = {
                'CSP': Pipeline([
                    ('csp', CSP(n_components=6, log=True, norm_trace=False)), 
                    ('svm', SVC(kernel='linear'))
                ]),
                'Geometry': Pipeline([
                    ('geom', GeometryFeatureExtractor()), 
                    ('s', StandardScaler()), 
                    ('svm', SVC(C=10))
                ]),
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
                # Fold별 표준편차도 저장하면 좋지만, 여기선 평균만 사용하고
                # 나중에 seaborn이 여러 피험자/반복 데이터를 이용해 CI를 그려줌.
                
                results.append({
                    'Subject': sub, 
                    'Scenario': scene,
                    'Noise Level ($\sigma$)': nl, 
                    'Method': name, 
                    'Accuracy (%)': avg_acc * 100
                })
            
            # 진행 상황 (간단히)
            # print(f"    Lv {nl:.1f} Done.")

# --- [시각화] Figure_f2-2 스타일 ---
df = pd.DataFrame(results)

# 시나리오 개수만큼 Subplot 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# 스타일 설정
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
palette = {'Hybrid': '#2ca02c', 'CSP': '#ff7f0e', 'Geometry': '#1f77b4'}
markers = {'Hybrid': 'o', 'CSP': 's', 'Geometry': 'D'}

titles = {
    'gaussian': 'Robustness to White Noise', 
    'eog': 'Robustness to EOG Artifacts', 
    'realistic': 'Real-world Noise Environment'
}

for i, scene in enumerate(scenarios):
    ax = axes[i]
    data_scene = df[df['Scenario'] == scene]
    
    # Lineplot with Error Bands (자동으로 신뢰구간 그려줌)
    sns.lineplot(
        data=data_scene, 
        x='Noise Level ($\sigma$)', 
        y='Accuracy (%)', 
        hue='Method', 
        style='Method',
        palette=palette,
        markers=markers,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        ax=ax
    )
    
    # Chance Level (33.3%)
    ax.axhline(33.33, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance')
    
    ax.set_title(titles[scene], fontweight='bold', pad=15)
    ax.set_ylim(20, 100)
    ax.set_xlabel('Noise Intensity')
    
    if i == 0:
        ax.set_ylabel('Accuracy (%)')
    else:
        ax.set_ylabel('')
        ax.legend_.remove() # 첫 번째 그래프에만 범례 표시

# 범례 정리 (첫 번째 그래프의 범례를 예쁘게)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles, labels=labels, title='Decoding Method', loc='lower left', frameon=True)

plt.tight_layout()
plt.show()

print("완료! 이제 촘촘한 곡선 그래프와 시나리오별 비교를 볼 수 있습니다.")