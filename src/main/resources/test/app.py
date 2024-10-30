import cv2
import numpy as np
from flask import Flask, Response, jsonify
import onnxruntime as ort
import requests
import torch
import torch.nn as nn
import os
from datetime import datetime

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
    input_frame = cv2.resize(frame, (224, 224)).astype(np.float32)
    input_frame = np.expand_dims(input_frame, axis=0)  # 배치 차원 추가
    input_frame = np.transpose(input_frame, (0, 3, 1, 2))  # ONNX가 기대하는 형식으로 변환

    # ONNX 모델 예측
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_frame})
    
    print("ONNX 모델 출력:", result)
    print("ONNX 모델 출력 형태:", [r.shape for r in result])

    # ONNX 모델 결과에서 눈, 코, 입 키포인트 제거
    keypoints = result[0].flatten()
    # 예를 들어, 눈, 코, 입의 인덱스가 0, 1, 2라고 가정
    filtered_keypoints = np.delete(keypoints, [0, 1, 2])

    # MLP 모델에 입력
    mlp_input = torch.tensor(filtered_keypoints).float()
    
    print("MLP 입력 크기:", mlp_input.shape)
    print("MLP 입력 내용:", mlp_input)
    
    if mlp_input.numel() == 0:
        print("경고: MLP 입력이 비어 있습니다.")
        return None, None  # 또는 적절한 기본값 반환

    with torch.no_grad():
        mlp_output = mlp_model(mlp_input)
    
    final_result = mlp_output.numpy()
    return final_result

def send_to_spring(prediction):
    url = 'http://localhost:8080/api/prediction'
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    elif isinstance(prediction, list):
        prediction = [item.tolist() if isinstance(item, np.ndarray) else item for item in prediction]
    data = {'prediction': prediction}
    response = requests.post(url, json=data)
    return response.status_code

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            prediction = process_frame(frame)
            if prediction is not None:
                print("예측 유형:", type(prediction))
                print("예측 내용:", prediction)
                
                status = send_to_spring(prediction)
                print("Spring 서버로 전송 상태:", status)
            
            # 프레임 인코딩 및 yield
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("프레임을 인코딩할 수 없습니다.")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            continue  # 오류가 발생해도 계속 진행

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
