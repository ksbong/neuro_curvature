import numpy as np

class GeometryExtractor:
    @staticmethod
    def calculate_curvature(z):
        """
        복소 궤적 z(t)로부터 기하학적 곡률 계산
        """
        x, y = np.real(z), np.imag(z)
        
        # 1차 및 2차 미분
        dx = np.gradient(x, axis=-1)
        dy = np.gradient(y, axis=-1)
        ddx = np.gradient(dx, axis=-1)
        ddy = np.gradient(dy, axis=-1)
        
        # Curvature 공식 적용
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5 + 1e-9
        
        return numerator / denominator