import numpy as np
from scipy.signal import savgol_filter

class GeometryExtractor:
    @staticmethod
    def calculate_curvature(z, window_length=11, polyorder=3):
        """
        [기존 기능] 곡률(Curvature) 계산
        공식: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        """
        x, y = np.real(z), np.imag(z)
        
        dx = savgol_filter(x, window_length, polyorder, deriv=1, axis=-1)
        dy = savgol_filter(y, window_length, polyorder, deriv=1, axis=-1)
        ddx = savgol_filter(x, window_length, polyorder, deriv=2, axis=-1)
        ddy = savgol_filter(y, window_length, polyorder, deriv=2, axis=-1)
        
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5 + 1e-9
        
        return numerator / denominator

    @staticmethod
    def calculate_basic_features(z, sfreq=160.0):
        """
        [추가] 힐베르트 변환 기본 정보: 순시 진폭, 위상, 주파수
        - Amplitude: 신호의 포락선(Envelope)
        - Phase: 현재 위상 각도
        - Frequency: 위상의 변화율 (각속도)
        """
        amplitude = np.abs(z)
        phase = np.angle(z)
        
        # 위상 풀기 (Unwrap) 후 미분하여 주파수 계산
        unwrapped_phase = np.unwrap(phase, axis=-1)
        # 차분(diff)을 이용해 주파수 근사 (Hz 단위 변환: * sfreq / 2pi)
        frequency = np.diff(unwrapped_phase, axis=-1, prepend=0) * (sfreq / (2 * np.pi))
        
        return amplitude, phase, frequency

    @staticmethod
    def calculate_complex_velocity(z, window_length=11, polyorder=3, sfreq=160.0):
        """
        [추가] 복소 평면상에서의 이동 속력 (Speed)
        v(t) = sqrt(x'(t)^2 + y'(t)^2)
        급격한 변화(Spike) 구간에서 속력이 높게 나옴.
        """
        x, y = np.real(z), np.imag(z)
        
        dx = savgol_filter(x, window_length, polyorder, deriv=1, axis=-1) * sfreq
        dy = savgol_filter(y, window_length, polyorder, deriv=1, axis=-1) * sfreq
        
        velocity = np.sqrt(dx**2 + dy**2)
        return velocity

    @staticmethod
    def calculate_torsion_3d(z, window_length=11, polyorder=4, sfreq=160.0):
        """
        [추가] 3차원 위상 공간 (Time, Real, Imag) 곡선의 비틀림(Torsion)
        
        곡선 r(t) = <t, x(t), y(t)> 라고 할 때,
        Torsion = (r' x r'') . r''' / |r' x r''|^2
        
        - 3계 미분까지 필요하므로 polyorder는 최소 4 이상이어야 함.
        """
        x, y = np.real(z), np.imag(z)
        
        # 1, 2, 3계 미분 (Time 축은 t이므로 1차 미분=1, 나머지=0)
        dx = savgol_filter(x, window_length, polyorder, deriv=1, axis=-1) * sfreq
        dy = savgol_filter(y, window_length, polyorder, deriv=1, axis=-1) * sfreq
        
        ddx = savgol_filter(x, window_length, polyorder, deriv=2, axis=-1) * (sfreq**2)
        ddy = savgol_filter(y, window_length, polyorder, deriv=2, axis=-1) * (sfreq**2)
        
        dddx = savgol_filter(x, window_length, polyorder, deriv=3, axis=-1) * (sfreq**3)
        dddy = savgol_filter(y, window_length, polyorder, deriv=3, axis=-1) * (sfreq**3)
        
        # r' = <1, dx, dy>
        # r'' = <0, ddx, ddy>
        # r''' = <0, dddx, dddy>
        
        # 외적 (r' x r'') 계산
        # i 성분: dx*ddy - dy*ddx
        # j 성분: -ddy
        # k 성분: ddx
        cross_x = dx * ddy - dy * ddx
        cross_y = -ddy
        cross_z = ddx
        
        # 분자: (r' x r'') . r'''
        # (cross_x * 0) + (cross_y * dddx) + (cross_z * dddy)
        numerator = cross_y * dddx + cross_z * dddy
        
        # 분모: |r' x r''|^2
        denominator = cross_x**2 + cross_y**2 + cross_z**2 + 1e-9
        
        torsion = numerator / denominator
        return torsion