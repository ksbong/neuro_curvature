# main.py
import numpy as np
import torch
import random

from src.train import run_training_pipeline
from src.utils import run_research_pipeline

def set_seed(seed=42):
    """실험 재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # 시드 고정 (중요!)
    set_seed(42)

    # ==========================================
    # [Mode 1] 기하학적 분석 & 시각화 (연구 단계)
    # : 3D 궤적, 곡률/비틀림 분포 등을 눈으로 확인
    # ==========================================
    run_research_pipeline()
    
    # ==========================================
    # [Mode 2] 정량적 비교 분석
    # : 두 조건(Left/Right) 간의 기하학적 차이 통계
    # ==========================================
    # run_quantitative_comparison()
    
    # ==========================================
    # [Mode 3] SNN 모델 학습
    # : 검증된 Feature로 실제 분류 모델 학습
    # ==========================================
    # run_training_pipeline(subjects=[1, 2, 3, 4, 5], epochs=50)