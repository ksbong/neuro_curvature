import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.signal import hilbert
from mne.decoding import CSP
from mne.time_frequency import tfr_multitaper
import mne

class_names = {0: 'Left Hand', 1: 'Right Hand', 2: 'Both Feet'}
class_colors = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c'}

def plot_manifold(X, y):
    print(" >> Plotting 3D Manifold...")
    z = hilbert(X, axis=-1)
    n_epochs, n_ch, n_times = z.shape
    X_flat = np.real(z).transpose(0, 2, 1).reshape(-1, n_ch)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_flat).reshape(n_epochs, n_times, 3)
    
    fig = go.Figure()
    for label_id in np.unique(y):
        indices = np.where(y == label_id)[0]
        if len(indices) > 0:
            idx = indices[0] # 대표 하나만 그리기
            traj = X_pca[idx]
            fig.add_trace(go.Scatter3d(
                x=traj[:,0], y=traj[:,1], z=traj[:,2], mode='lines',
                name=class_names[label_id],
                line=dict(color=class_colors[label_id], width=5), opacity=0.9
            ))
    fig.update_layout(title="Neural Manifold (3-Class)", width=800, height=600)
    fig.show()

def plot_feature_distribution(features, y):
    print(" >> Plotting Feature Distribution...")
    df = pd.DataFrame({
        'Curvature': features[:, 0],
        'Tangling': features[:, 2],
        'Condition': [class_names[v] for v in y]
    })
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.violinplot(data=df, x='Condition', y='Curvature', palette='muted')
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='Condition', y='Tangling', palette='muted')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(X, y, info):
    print(" >> Plotting Spectrogram...")
    # Info 객체를 이용해 임시 Epochs 생성
    epochs = mne.EpochsArray(X, info, events=np.column_stack((np.arange(len(y)), np.zeros(len(y), int), y)), tmin=0.5, verbose=False)
    
    freqs = np.arange(8, 30, 1)
    power = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs/2., use_fft=True, return_itc=False, average=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['C3', 'Cz', 'C4']
    titles = ['C3 (Right Hand)', 'Cz (Feet)', 'C4 (Left Hand)']
    
    for i, ch in enumerate(channels):
        if ch in epochs.ch_names:
            power.plot([epochs.ch_names.index(ch)], baseline=None, mode='logratio', 
                       axes=ax[i], show=False, colorbar=True)
            ax[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()