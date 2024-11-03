import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from collections import Counter
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
import httpx
import asyncio

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

class MLP(nn.Module):
    def __init__(self, input_size, f1_num, f2_num, f3_num, f4_num, f5_num, f6_num, d1, d2, d3, d4, d5, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, f1_num)
        self.fc2 = nn.Linear(f1_num, f2_num)
        self.fc3 = nn.Linear(f2_num, f3_num)
        self.fc4 = nn.Linear(f3_num, f4_num)
        self.fc5 = nn.Linear(f4_num, f5_num)
        self.fc6 = nn.Linear(f5_num, f6_num)
        self.fc7 = nn.Linear(f6_num, num_classes)
        self.dropout1 = nn.Dropout(p=d1)
        self.dropout2 = nn.Dropout(p=d2)
        self.dropout3 = nn.Dropout(p=d3)
        self.dropout4 = nn.Dropout(p=d4)
        self.dropout5 = nn.Dropout(p=d5)

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

model = MLP(12, 64, 128, 256, 256, 128, 64, 0.2, 0.2, 0.2, 0.2, 0.2, 6)
model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth'))
first_mlp_model = model.to(device).eval()

First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

Label_List = []
nNotDetected = 0
boolHumanCheck = False

async def send_to_spring(mode):
    spring_boot_url = f"http://localhost:8080/api/mode/{mode}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(spring_boot_url)
            if response.status_code == 200:
                print(f"모드 {mode} 전송 성공")
            else:
                print(f"모드 전송 실패: {response.status_code}")
        except httpx.RequestError as e:
            print(f"모드 전송 중 오류 발생: {e}")

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def make_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
    else:
        slope = 0
    return slope

async def generate_frames(mode):
    global Label_List, nNotDetected, boolHumanCheck
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        with torch.no_grad():
            outputs = keypoint_model(input_tensor)

        output = outputs[0]
        scores = output['scores'].cpu().numpy()
        high_scores_idx = np.where(scores > 0.95)[0]

        display_frame = frame.copy()

        for idx in high_scores_idx:
            keypoints = output['keypoints'][idx].cpu().numpy()
            keypoint_scores = output['keypoints_scores'][idx].cpu().numpy()
            boxes = output['boxes'][idx].cpu().numpy()

            check_count = sum(1 for kp_score in keypoint_scores if kp_score < 0.9)

            if check_count < 2:
                angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                ]]

                angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = first_mlp_model(angles_tensor)
                    _, predicted_label = torch.max(prediction, 1)
                First_Label = First_MLP_label_map[predicted_label.item()]

                if len(Label_List) >= 10:
                    Label_List.pop(0)
                Label_List.append(predicted_label.item())

                if len(Label_List) >= 10:
                    if Label_List[9] == 0:
                        await send_to_spring(mode)  # 모드에 따라 스프링 부트로 전송
                        Label_List.clear()
                    elif 1 in Label_List and nNotDetected >= 4:
                        await send_to_spring(mode)  # 모드에 따라 스프링 부트로 전송
                        Label_List.clear()

                x1, y1, x2, y2 = map(int, boxes)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, First_Label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                boolHumanCheck = True
                nNotDetected = 0
            else:
                boolHumanCheck = False
                if nNotDetected < 5:
                    nNotDetected += 1

        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
async def index():
    return Response(content="""<html><head><title>실시간 영상</title></head><body><h1>실시간 영상</h1><img src="/video_feed" width="640" height="480"></body></html>""", media_type="text/html")

@app.get("/video_feed")
async def video_feed(mode: int):
    return StreamingResponse(generate_frames(mode), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

cap.release()
cv2.destroyAllWindows()