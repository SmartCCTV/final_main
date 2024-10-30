from fastapi import FastAPI
import torch
import torch.nn as nn

# MLP 모델 정의 (여기서는 간단한 예시)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 6)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc7(out)
        return out


# FastAPI 앱 생성
app = FastAPI()

# 사전 훈련된 모델 로드
model = MLP()
model.load_state_dict(torch.load("Bottom_Loss_Validation_MLP.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

@app.post("/predict/")
async def predict(data: dict):
    # 입력 데이터를 텐서로 변환
    input_data = torch.tensor(data['inputs'])
    # 모델 추론
    with torch.no_grad():
        prediction = model(input_data).item()
    return {"prediction": prediction}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return {"message": "No favicon available"}

# 기존의 /predict/ 엔드포인트도 포함
