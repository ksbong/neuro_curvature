import mne
from mne.datasets import eegbci
import numpy as np

def load_multiclass_data(subject, runs_hand=[4, 8, 12], runs_feet=[6, 10, 14]):
    # 1. 손 데이터 로드 (Left=2, Right=3)
    raw_fnames_hand = eegbci.load_data(subject, runs_hand, verbose=False)
    raw_hand = mne.io.read_raw_edf(raw_fnames_hand[0], preload=True, verbose=False)
    for f in raw_fnames_hand[1:]:
        raw_hand.append(mne.io.read_raw_edf(f, preload=True, verbose=False))
    
    # 2. 발 데이터 로드 (Fists=2, Feet=3 -> Feet만 쓸 예정)
    raw_fnames_feet = eegbci.load_data(subject, runs_feet, verbose=False)
    raw_feet = mne.io.read_raw_edf(raw_fnames_feet[0], preload=True, verbose=False)
    for f in raw_fnames_feet[1:]:
        raw_feet.append(mne.io.read_raw_edf(f, preload=True, verbose=False))

    # 3. 공통 전처리
    picks = None
    for raw in [raw_hand, raw_feet]:
        eegbci.standardize(raw)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        raw.filter(8., 30., fir_design='firwin', verbose=False)
        # 채널 선택 (C3, Cz, C4 등 운동 영역 위주)
        if picks is None:
            picks = mne.pick_channels(raw.ch_names, ['C3', 'Cz', 'C4', 'FC3', 'FC4', 'CP3', 'CP4'])

    # 4. 이벤트 ID 재매핑
    # Hand: T1(2)->0(Left), T2(3)->1(Right)
    events_h, _ = mne.events_from_annotations(raw_hand, event_id=dict(T1=2, T2=3), verbose=False)
    events_h[:, -1] -= 2 

    # Feet: T2(3)->2(Feet)
    events_f, _ = mne.events_from_annotations(raw_feet, event_id=dict(T2=3), verbose=False)
    events_f[:, -1] = 2

    # 5. Epoch 생성
    epochs_h = mne.Epochs(raw_hand, events_h, event_id={'Left':0, 'Right':1}, 
                          tmin=0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose=False)
    epochs_f = mne.Epochs(raw_feet, events_f, event_id={'Feet':2}, 
                          tmin=0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose=False)

    # 6. 병합
    epochs_all = mne.concatenate_epochs([epochs_h, epochs_f])
    
    # [수정됨] X, y 딱 두 개만 반환하도록 수정
    return epochs_all.get_data(), epochs_all.events[:, -1]