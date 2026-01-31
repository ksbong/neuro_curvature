import torch
import mne
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from src.core.preprocess import EEGLoader
from src.core.geometry import GeometryExtractor
from src.data.encoder import MultiFeatureEncoder
from src.data.dataset import GeometricEEGDataset
from src.models.snn import DynamicDeepSNN

def run_training_pipeline(subjects=[1, 2, 3, 4, 5], epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = EEGLoader()
    raw = loader.fetch_and_load(subjects=subjects)
    
    # 데이터 준비 및 표준화
    raw_eeg = raw.get_data()
    raw_eeg = (raw_eeg - np.mean(raw_eeg)) / np.std(raw_eeg)
    
    analytic_data, _ = loader.process_to_analytic(raw)
    curvature_log = np.log1p(GeometryExtractor.calculate_curvature(analytic_data))
    
    # 멀티 피처 인코딩
    encoder = MultiFeatureEncoder()
    spikes = encoder.encode(curvature_log, raw_eeg)
    
    dataset = GeometricEEGDataset(spikes, mne.events_from_annotations(raw)[0], 
                                mne.events_from_annotations(raw)[1], window_size=480)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = DynamicDeepSNN(num_inputs=spikes.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1.2, 1.2]).to(device))

    print(f"Starting training on {device}... Input Dim: {spikes.shape[0]}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.transpose(0, 1).to(device), target.to(device)
            optimizer.zero_grad()
            spk_out, _ = model(data)
            loss = criterion(torch.mean(spk_out, dim=0), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        all_targets, all_preds = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.transpose(0, 1).to(device)
                _, mem_out = model(data)
                all_targets.extend(target.numpy())
                all_preds.extend(mem_out.argmax(dim=1).cpu().numpy())

        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]) #
        acc = 100. * np.sum(np.diag(cm)) / np.sum(cm)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
        if (epoch + 1) % 5 == 0: print(f"CM:\n{cm}")