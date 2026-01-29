import torch
from torch.utils.data import DataLoader, random_split
from src.core.preprocess import EEGLoader
from src.core.geometry import GeometryExtractor
from src.data.encoder import GeometricSpikeEncoder
from src.data.dataset import GeometricEEGDataset
from src.models.snn import GeometricSNN

def run_training_pipeline(subjects=[1, 2, 3, 4, 5], epochs=50, batch_size=32, lr=5e-4):
    print("\n--- Phase 4: Multi-Subject SNN Training ---")
    
    # 1. 데이터 로드 및 전처리
    loader = EEGLoader()
    raw = loader.fetch_and_load(subjects=subjects) 
    analytic_data, _ = loader.process_to_analytic(raw)
    
    # 2. 특징 추출 및 인코딩
    curvature = GeometryExtractor.calculate_curvature(analytic_data)
    encoder = GeometricSpikeEncoder(threshold_type='std')
    spikes, _ = encoder.threshold_encoding(curvature)
    
    # 3. 데이터셋 생성
    import mne
    events, event_id = mne.events_from_annotations(raw)
    dataset = GeometricEEGDataset(spikes, events, event_id, window_size=480) 
    
    # 4. 분할 및 로더
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometricSNN(num_inputs=spikes.shape[0], num_hidden=128, num_outputs=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 6. Training Loop
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data = data.transpose(0, 1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            spk_out, _ = model(data)
            output = torch.mean(spk_out, dim=0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 평가
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.transpose(0, 1).to(device)
                spk_out, _ = model(data)
                output = torch.mean(spk_out, dim=0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.to(device).view_as(pred)).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Test Acc: {accuracy:.2f}%")

    print("--- Training Execution Finished ---")
    return model    