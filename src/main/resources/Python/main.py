import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from collections import Counter
from fastapi import FastAPI, Response, Query
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio
import base64
import os
from datetime import datetime
import httpx

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

async def send_label_and_image_to_spring(label, frame, max_retries=3):
    spring_boot_url = "http://localhost:8080/api/send-label-and-image"

    for attempt in range(max_retries):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                "label": label,
                "image": image_base64
            }

            print(f"전송 시도 {attempt + 1}/{max_retries}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(spring_boot_url, json=data)

                if response.status_code == 200:
                    print(f"라벨과 이미지 전송 성공")
                    return True
                else:
                    print(f"전송 실패 (상태 코드: {response.status_code})")
                    print(f"응답 내용: {response.text}")

        except Exception as e:
            print(f"전송 시도 {attempt + 1} 실패: {str(e)}")
            if attempt < max_retries - 1:
                print(f"{2 ** attempt}초 후 재시도...")
                await asyncio.sleep(2 ** attempt)
            continue

    print("최대 재시도 횟수 초과")
    return False

def show_message():
    print("경고", "낙상이 감지 되었습니다!")

def draw_keypoints_and_connections(background, keypoints):
    cnt = 0
    for point in keypoints:
        if cnt > 5:
            x, y = map(int, point[:2])
            cv2.circle(background, (x, y), 5, (0, 0, 255), -1)
        cnt += 1

    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for connection in connections:
        start_point = tuple(map(int, keypoints[connection[0]][:2]))
        end_point = tuple(map(int, keypoints[connection[1]][:2]))
        cv2.line(background, start_point, end_point, (0, 255, 0), 2)

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
        out = self.dropout5(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        return out

first_mlp_model = MLP(12, 64, 128, 256, 256, 128, 64, 0.2, 0.2, 0.2, 0.2, 0.2, 6)
first_mlp_model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth'))
first_mlp_model = first_mlp_model.to(device).eval()

First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

current_mode = "1"
Label_List = []
nNotDetected = 0
boolHumanCheck = False

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

def reset_state():
    global Label_List, nNotDetected, boolHumanCheck
    Label_List = []
    nNotDetected = 0
    boolHumanCheck = False

reset_state()

async def generate_frames():
    global Label_List, boolHumanCheck, nNotDetected
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다. 1초 후 다시 시도합니다.")
            await asyncio.sleep(1)
            continue

        input_tensor = preprocess(frame)
        with torch.no_grad():
            outputs = keypoint_model(input_tensor)

        if current_mode in ['3', '4']:
            display_frame = np.zeros_like(frame)

        if current_mode == "1":
            output = outputs[0]
            scores = output['scores'].cpu().numpy()
            high_scores_idx = np.where(scores > 0.95)[0]

            if len(high_scores_idx) > 0:
                keypoints = output['keypoints'][high_scores_idx[0]].cpu().numpy()
                keypoint_scores = output['keypoints_scores'][high_scores_idx[0]].cpu().numpy()
                boxes = output['boxes'][high_scores_idx[0]].cpu().numpy()

                if sum(keypoint_scores < 0.9) < 2:
                    angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                    ]]

                    angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = first_mlp_model(angles_tensor)
                        predicted_label = torch.max(prediction, 1)[1].item()

                    First_Label = First_MLP_label_map[predicted_label]

                    Label_List.append(predicted_label)
                    if len(Label_List) > 10:
                        Label_List.pop(0)

                    if len(Label_List) == 10:
                        if Label_List[-1] == 0:
                            counterBefore = Counter(Label_List)
                            most_common_count_Before = counterBefore.most_common(1)[0][1]
                            counterBeforeLabel = counterBefore.most_common(1)[0][0]

                            if most_common_count_Before >= 7 and (counterBeforeLabel == 1 or counterBeforeLabel == 4):
                                box_color = (0, 0, 255)
                                show_message()
                                success = await send_label_and_image_to_spring(First_Label, frame)
                                if not success:
                                    print("위험 감지 알림 전송 실패")
                                else:
                                    print("위험 감지 알림 전송 성공")
                        else:
                            box_color = (0, 255, 0)

                        x1, y1, x2, y2 = map(int, boxes)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        (label_width, label_height), baseline = cv2.getTextSize(First_Label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                                1, 2)
                        cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), box_color,
                                      cv2.FILLED)
                        cv2.putText(frame, First_Label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)

                        boolHumanCheck = True
                        nNotDetected = 0
                else:
                    boolHumanCheck = False
                    nNotDetected = min(nNotDetected + 1, 5)

        elif current_mode == "2":
            for idx in range(len(outputs)):
                output = outputs[idx]
                scores = output['scores'].cpu().numpy()
                high_scores_idx = np.where(scores > 0.95)[0]

                for high_idx in high_scores_idx:
                    keypoints = output['keypoints'][high_idx].cpu().numpy()
                    keypoint_scores = output['keypoints_scores'][high_idx].cpu().numpy()
                    boxes = output['boxes'][high_idx].cpu().numpy()

                    if sum(keypoint_scores < 0.9) < 2:
                        angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                        ]]

                        angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            prediction = first_mlp_model(angles_tensor)
                            predicted_label = torch.max(prediction, 1)[1].item()

                        First_Label = First_MLP_label_map[predicted_label]

                        Label_List.append(predicted_label)
                        if len(Label_List) > 10:
                            Label_List.pop(0)

                        if len(Label_List) == 10:
                            if Label_List[-1] == 0:
                                counterBefore = Counter(Label_List)
                                most_common_count_Before = counterBefore.most_common(1)[0][1]
                                counterBeforeLabel = counterBefore.most_common(1)[0][0]

                                if most_common_count_Before >= 7 and (counterBeforeLabel == 1 or counterBeforeLabel == 4):
                                    box_color = (0, 0, 255)
                                    show_message()
                                    success = await send_label_and_image_to_spring(First_Label, frame)
                                    if not success:
                                        print("위험 감지 알림 전송 실패")
                                    else:
                                        print("위험 감지 알림 전송 성공")

                        x1, y1, x2, y2 = map(int, boxes)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (label_width, label_height), baseline = cv2.getTextSize(First_Label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                                1, 2)
                        cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0),
                                      cv2.FILLED)
                        cv2.putText(frame, First_Label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)

                        boolHumanCheck = True
                        nNotDetected = 0
                    else:
                        boolHumanCheck = False
                        nNotDetected = min(nNotDetected + 1, 5)

        elif current_mode == "3":
            output = outputs[0]
            scores = output['scores'].cpu().numpy()
            high_scores_idx = np.where(scores > 0.95)[0]

            if len(high_scores_idx) > 0:
                keypoints = output['keypoints'][high_scores_idx[0]].cpu().numpy()
                keypoint_scores = output['keypoints_scores'][high_scores_idx[0]].cpu().numpy()

                if sum(keypoint_scores < 0.9) < 2:
                    draw_keypoints_and_connections(frame, keypoints)

                    angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                    ]]
                    angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = first_mlp_model(angles_tensor)
                        predicted_label = torch.max(prediction, 1)[1].item()

                    First_Label = First_MLP_label_map[predicted_label]

                    if predicted_label == 0:
                        show_message()
                        await send_label_and_image_to_spring(First_Label, frame)

        elif current_mode == "4":
            for idx in range(len(outputs)):
                output = outputs[idx]
                scores = output['scores'].cpu().numpy()
                high_scores_idx = np.where(scores > 0.95)[0]

                for high_idx in high_scores_idx:
                    keypoints = output['keypoints'][high_idx].cpu().numpy()
                    keypoint_scores = output['keypoints_scores'][high_idx].cpu().numpy()

                    if sum(keypoint_scores < 0.9) < 2:
                        draw_keypoints_and_connections(frame, keypoints)

                        angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
                        ]]
                        angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            prediction = first_mlp_model(angles_tensor)
                            predicted_label = torch.max(prediction, 1)[1].item()

                        First_Label = First_MLP_label_map[predicted_label]

                        if predicted_label == 0:
                            show_message()
                            await send_label_and_image_to_spring(First_Label, frame)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
async def index():
    return Response(content="""<html><head><title>실시간 영상</title></head><body><h1>실시간 영상</h1><img src="/video_feed" width="640" height="480"></body></html>""", media_type="text/html")

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/set_mode")
async def set_mode(mode: str = Query(...)):
    global current_mode
    current_mode = mode
    reset_state()
    print(f"현재 모드: {current_mode}")
    return {"message": f"{current_mode} 모드로 설정되었습니다."}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

cap.release()
cv2.destroyAllWindows()