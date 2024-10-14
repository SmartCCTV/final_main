# model.py
import numpy as np

class SimpleModel:
    def predict(self, input_data):
        # 간단한 예측 로직 (입력값의 두 배를 반환)
        return np.array(input_data) * 2
