import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.datasets import eegbci
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from scipy.signal import savgol_filter, hilbert
import matplotlib.pyplot as plt

# ==================================================================
# 1. Feature Extractor
# ==================================================================
class MotorCortexGeoFeatures:
    def __init__(self, ref_L, ref_R):
        self.cov_est = Covariances(estimator='lwf')
        self.ref_L = ref_L
        self.ref_R = ref_R

    def compute_features(self, trial_data):
        z = hilbert(trial_data, axis=1) 
        traj_phase = np.concatenate([z.real, z.imag], axis=0)
        
        d_L, d_R = self._compute_dual_riemann(traj_phase)
        k_t, Q_t = self._compute_curvature(traj_phase)
        
        target_len = len(d_L)
        if target_len > 0:
            k_t = np.interp(np.linspace(0, len(k_t), target_len), np.arange(len(k_t)), k_t)
            Q_t = np.interp(np.linspace(0, len(Q_t), target_len), np.arange(len(Q_t)), Q_t)
            return np.stack([d_L, d_R, k_t, Q_t], axis=1)
        return None

    def _compute_dual_riemann(self, trajectory):
        window = 50; stride = 10
        n_wins = (trajectory.shape[1] - window) // stride
        if n_wins <= 0: return np.array([]), np.array([])

        segments = []
        for w in range(n_wins):
            s = w * stride
            segments.append(trajectory[:, s:s+window])
        
        covs = self.cov_est.fit_transform(np.array(segments))
        dim = covs.shape[1]
        stab = np.eye(dim) * 1e-6
        
        d_L_list, d_R_list = [], []
        for C in covs:
            try:
                C_stab = C + stab
                dist_L = distance_riemann(C_stab, self.ref_L + stab)
                dist_R = distance_riemann(C_stab, self.ref_R + stab)
            except:
                dist_L, dist_R = 0.0, 0.0
            d_L_list.append(dist_L)
            d_R_list.append(dist_R)
            
        return np.array(d_L_list), np.array(d_R_list)

    def _compute_curvature(self, trajectory):
        try: traj_smooth = savgol_filter(trajectory, 11, 3, axis=1)
        except: traj_smooth = trajectory
        
        dt = 1.0
        v = np.gradient(traj_smooth, axis=1) / dt
        a = np.gradient(v, axis=1) / dt
        
        v_sq = np.sum(v**2, axis=0)
        a_sq = np.sum(a**2, axis=0)
        dot = np.sum(v*a, axis=0)
        
        numer = np.sqrt(np.maximum(0, v_sq * a_sq - dot**2))
        denom = (v_sq ** 1.5) + 1e-9
        
        k = numer / denom
        Q = np.sqrt(a_sq) / (np.sqrt(v_sq) + 1e-6)
        
        return np.log1p(k), np.log1p(Q)

# ==================================================================
# 2. SNN Model
# ==================================================================
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=25.0):
        ctx.scale = scale 
        ctx.save_for_backward(input)
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * ctx.scale / ((1 + ctx.scale * input.abs())**2), None

class ThresholdEncodingLayer(nn.Module):
    def __init__(self, n_thresholds=32):
        super().__init__()
        self.thresholds = torch.linspace(-3.0, 3.0, n_thresholds)
        self.sigma = 5.0 / n_thresholds 
        
    def forward(self, x):
        t = self.thresholds.to(x.device).view(1, 1, -1)
        return torch.exp(-0.5 * ((x - t)**2) / (self.sigma**2))

class NeuroMotorSNN(nn.Module):
    def __init__(self, n_thresholds=32, hidden=128):
        super().__init__()
        self.enc_dL = ThresholdEncodingLayer(n_thresholds)
        self.enc_dR = ThresholdEncodingLayer(n_thresholds)
        self.enc_k  = ThresholdEncodingLayer(n_thresholds)
        self.enc_Q  = ThresholdEncodingLayer(n_thresholds)
        
        input_size = 4 * n_thresholds
        
        self.fc_in = nn.Linear(input_size, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.5) 
        self.fc_out = nn.Linear(hidden, 2) 
        
        self.beta = 0.9 
        self.thresh = 0.5 
        self.act = FastSigmoid.apply
        
    def forward(self, x):
        dL = x[:,:,0].unsqueeze(-1)
        dR = x[:,:,1].unsqueeze(-1)
        k  = x[:,:,2].unsqueeze(-1)
        Q  = x[:,:,3].unsqueeze(-1)
        
        spikes_in = torch.cat([
            self.enc_dL(dL), self.enc_dR(dR), 
            self.enc_k(k), self.enc_Q(Q)
        ], dim=2)
        
        mem = torch.zeros(x.size(0), 128).to(x.device)
        readout = 0
        
        for t in range(x.size(1)):
            curr = self.ln(self.fc_in(spikes_in[:, t]))
            curr = self.dropout(curr)
            mem = mem * self.beta + curr * (1 - self.beta)
            spk = self.act(mem - self.thresh)
            mem = mem - spk * self.thresh
            readout = readout + self.fc_out(spk)
            
        return readout, None

# ==================================================================
# 3. Main Routine
# ==================================================================
def main_final_neuro():
    print(">>> [Setup] Loading Subjects 1~10...")
    subjects = list(range(1, 11)) 
    runs = [4, 8, 12] 
    
    # [Target] C3, Cz, C4 (대소문자 무관하게 처리할 예정)
    target_channels = ['C3', 'Cz', 'C4']
    
    all_features, all_labels = [], []
    
    for s in subjects:
        try:
            f = eegbci.load_data(s, runs, update_path=False, verbose=False)
            raws = [mne.io.read_raw_edf(fi, preload=True, verbose=False) for fi in f]
            raw = mne.io.concatenate_raws(raws)
            
            # ---------------------------------------------------------
            # [FIX] Channel Name Cleaning (Strip dots!)
            # ---------------------------------------------------------
            # C3.., C3. -> C3 로 통일
            raw.rename_channels(lambda x: x.strip('.'))
            
            # 이제 안전하게 Pick 가능
            # (channel 이름이 조금 달라도 에러 안 나게, 존재하는 것만 선택)
            available_targets = [ch for ch in target_channels if ch in raw.ch_names]
            
            if len(available_targets) < 3:
                # 혹시라도 대문자 이슈일 수 있으니 (CZ -> Cz) 체크
                mapping = {}
                for ch in raw.ch_names:
                    if ch.upper() == 'C3': mapping[ch] = 'C3'
                    elif ch.upper() == 'C4': mapping[ch] = 'C4'
                    elif ch.upper() == 'CZ': mapping[ch] = 'Cz'
                raw.rename_channels(mapping)
                available_targets = [ch for ch in target_channels if ch in raw.ch_names]
            
            if len(available_targets) == 0:
                print(f"   > Subject {s}: No motor channels found. Skipping.")
                continue
                
            raw.pick(available_targets)
            
            # Filter
            raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, event_id={'T1':0, 'T2':1}, verbose=False)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=4, baseline=None, preload=True, verbose=False)
            
            data = epochs.get_data(copy=True) * 1e6 
            y = epochs.events[:, -1]
            
            # Calibration (Class Means)
            z = hilbert(data, axis=2)
            traj = np.concatenate([z.real, z.imag], axis=1) 
            
            traj_L = traj[y == 0]
            traj_R = traj[y == 1]
            
            # Helper for covariance
            def get_covs(trajectory_set):
                c_list = []
                for t in trajectory_set:
                    if np.std(t) < 1e-9: t += np.random.randn(*t.shape)*1e-6
                    c_list.append(np.cov(t))
                return np.array(c_list)

            # Check if we have both classes
            if len(traj_L) == 0 or len(traj_R) == 0:
                continue

            mean_L = mean_riemann(get_covs(traj_L))
            mean_R = mean_riemann(get_covs(traj_R))
            
            # Extract
            engine = MotorCortexGeoFeatures(mean_L, mean_R)
            subj_feats = []
            valid_idx = []
            
            print(f"   > Subject {s}: Extracting Geometry...", end='\r')
            
            for i in range(len(data)):
                f = engine.compute_features(data[i])
                if f is not None:
                    subj_feats.append(f)
                    valid_idx.append(i)
            
            if not subj_feats: continue
            
            subj_feats = np.array(subj_feats, dtype=np.float32)
            subj_y = y[valid_idx]
            
            # Z-Score
            for f_idx in range(4):
                m = np.mean(subj_feats[:, :, f_idx])
                std = np.std(subj_feats[:, :, f_idx]) + 1e-6
                subj_feats[:, :, f_idx] = (subj_feats[:, :, f_idx] - m) / std
                subj_feats[:, :, f_idx] = np.clip(subj_feats[:, :, f_idx], -3, 3)
                
            all_features.append(subj_feats)
            all_labels.append(subj_y)
            
        except Exception as e:
            print(f"\nError Subject {s}: {e}")
            
    if not all_features:
        print("\n>>> CRITICAL: No valid data loaded.")
        return

    X = np.concatenate(all_features, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    print(f"\n>>> Final Dataset: {X.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.tensor(X).to(device), torch.tensor(Y, dtype=torch.long).to(device))
    
    train_len = int(0.85 * len(ds))
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, len(ds)-train_len])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    model = NeuroMotorSNN(n_thresholds=32, hidden=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss()
    
    print("\n>>> START TRAINING (C3/Cz/C4 + 8-30Hz Filtered)...")
    train_hist, test_hist = [], []
    
    for epoch in range(60):
        model.train()
        cor, tot = 0, 0
        for inp, tar in train_loader:
            optimizer.zero_grad()
            out, _ = model(inp)
            loss = criterion(out, tar)
            loss.backward()
            optimizer.step()
            cor += (torch.max(out, 1)[1] == tar).sum().item()
            tot += tar.size(0)
        
        train_acc = 100*cor/tot
        train_hist.append(train_acc)
        
        model.eval()
        c_t, t_t = 0, 0
        with torch.no_grad():
            for i_t, r_t in test_loader:
                o_t, _ = model(i_t)
                c_t += (torch.max(o_t, 1)[1] == r_t).sum().item()
                t_t += r_t.size(0)
        test_acc = 100*c_t/t_t
        test_hist.append(test_acc)
        
        if epoch % 5 == 0:
            print(f"Ep {epoch:02d} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
            
    print(f"\n>>> FINAL TEST ACCURACY: {test_acc:.2f}%")
    plt.plot(train_hist, label='Train')
    plt.plot(test_hist, label='Test')
    plt.title("Neuro-Geometric SNN (Filtered)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_final_neuro()