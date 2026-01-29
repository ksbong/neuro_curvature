from src.train import run_training_pipeline
from src.utils import run_research_pipeline, run_quantitative_comparison

if __name__ == "__main__":
    # 1. 기초 연구 및 시각화가 필요할 때
    # run_research_pipeline()
    # run_quantitative_comparison()
    
    # 2. SNN 학습 실행
    run_training_pipeline(subjects=[1, 2, 3, 4, 5], epochs=50)