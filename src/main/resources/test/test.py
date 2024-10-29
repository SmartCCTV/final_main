import cv2
import numpy as np
from flask import Flask, Response, jsonify
import onnxruntime as ort
import requests
import torch
import torch.nn as nn

app = Flask(__name__)

# ONNX 모델 로드
model_path = 'keypointrcnn_cpu.onnx'
session = ort.InferenceSession(model_path)

# MLP 모델 정의 및 로드
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)  # 256에서 128로 변경
        self.fc6 = nn.Linear(128, 64)   # 입력 크기를 256에서 128로 변경
        self.fc7 = nn.Linear(64, 6)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

# 모델 초기화
mlp_model = MLP(input_size=12, hidden_size=64, output_size=6)

# 가중치 로드
mlp_model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth', map_location=torch.device('cpu'), weights_only=True))
mlp_model.eval()

def process_frame(frame):
    # 프레임 전처리 (모델에 맞게 조정 필요)
    # 예시: 프레임을 리사이즈하고 float32로 변환
    input_frame = cv2.resize(frame, (224, 224)).astype(np.float32)
    input_frame = np.expand_dims(input_frame, axis=0)  # 배치 차원 추가
    input_frame = np.transpose(input_frame, (0, 3, 1, 2))  # ONNX가 기대하는 형식으로 변환

    # ONNX 모델 예측
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_frame})

    # ONNX 모델 결과를 MLP 모델에 입력
    mlp_input = torch.tensor(result[0].flatten()).float()
    with torch.no_grad():
        mlp_output = mlp_model(mlp_input)
    
    final_result = mlp_output.numpy()

    return final_result  # MLP 모델을 통과한 최종 결과

def send_to_spring(prediction):
    # Spring 서버로 결과 전송
    url = 'http://your_spring_server_address/api/prediction'
    data = {'prediction': prediction}  # numpy 배열을 리스트로 변환
    response = requests.post(url, json=data)
    return response.status_code

@app.route('/stream')
def stream():
    # 웹캠을 열어 스트림 처리
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리 및 모델 적용
        prediction = process_frame(frame)

        # 결과를 Spring 서버로 전송
        send_to_spring(prediction)

        # 원본 프레임을 그대로 송출
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
