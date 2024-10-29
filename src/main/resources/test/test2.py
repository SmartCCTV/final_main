import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# ONNX 모델 로드
model_path = 'keypointrcnn_cpu.onnx'
# CUDA를 사용하는 ONNX Runtime 세션 초기화
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# MLP 모델 정의 및 로드
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
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
mlp_model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth', map_location=torch.device('cuda')))
mlp_model.eval()

def process_frame(frame):
    input_frame = cv2.resize(frame, (224, 224)).astype(np.float32)
    input_frame = np.expand_dims(input_frame, axis=0)
    input_frame = np.transpose(input_frame, (0, 3, 1, 2))

    # ONNX 모델 예측
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_frame})
    
    print("ONNX 모델 출력:", result)
    print("ONNX 모델 출력 형태:", [r.shape for r in result])

    # MLP 입력을 GPU로 옮김
    mlp_input = torch.tensor(result[0].flatten()).float().to('cuda')
    
    print("MLP 입력 크기:", mlp_input.shape)
    print("MLP 입력 내용:", mlp_input)
    
    if mlp_input.numel() == 0:
        print("경고: MLP 입력이 비어 있습니다.")
        return None

    with torch.no_grad():
        mlp_output = mlp_model(mlp_input)
    
    final_result = mlp_output.cpu().numpy()  # 결과를 CPU로 옮겨서 넘파이 배열로 변환

    return final_result

def main():
    cap = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Frame', frame)
            else:
                print("프레임을 읽을 수 없습니다.")
            
            prediction = process_frame(frame)
            if prediction is not None:
                print("예측 유형:", type(prediction))
                print("예측 내용:", prediction)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
