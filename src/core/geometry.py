import numpy as np
from scipy.signal import savgol_filter

class GeometryExtractor:
    @staticmethod
    def calculate_curvature(z, window_length=11, polyorder=3):
        """
        Savitzky-Golay 필터를 사용하여 부드러운 미분값을 얻고 곡률을 계산함.
        """
        x, y = np.real(z), np.imag(z)
        
        # 1차 및 2차 미분값 (Savgol 필터 활용)
        dx = savgol_filter(x, window_length, polyorder, deriv=1, axis=-1)
        dy = savgol_filter(y, window_length, polyorder, deriv=1, axis=-1)
        ddx = savgol_filter(x, window_length, polyorder, deriv=2, axis=-1)
        ddy = savgol_filter(y, window_length, polyorder, deriv=2, axis=-1)
        
        # 곡률 공식: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5 + 1e-9
        
        curvature = numerator / denominator
        return curvature