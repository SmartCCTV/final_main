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
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
from datetime import datetime
import asyncio

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def show_message():
    print("경고: 낙상이 감지 되었습니다!")


def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.relu(self.fc2(out))
        out = self.dropout3(out)
        out = self.relu(self.fc3(out))
        out = self.dropout4(out)
        out = self.relu(self.fc4(out))
        out = self.dropout3(out)
        out = self.relu(self.fc5(out))
        out = self.dropout2(out)
        out = self.relu(self.fc6(out))
        out = self.dropout1(out)
        out = self.fc7(out)
        return out


first_mlp_model = MLP(input_size=12, num_classes=6)
first_mlp_model.load_state_dict(torch.load('C:/Users/PC1/PycharmProjects/PythonProject/Bottom_Loss_Validation_MLP.pth'))
first_mlp_model = first_mlp_model.to(device).eval()


def make_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        return (point1[1] - point2[1]) / (point1[0] - point2[0])
    return 0


First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

Label_List = []
boolHumanCheck = False
nNotDetected = 0


async def send_label_to_spring(label):
    spring_boot_url = "http://localhost:8080/api/send-label-and-image"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(spring_boot_url, json={"label": label})
            if response.status_code == 200:
                print(f"라벨 '{label}' 전송 성공")
            else:
                print(f"라벨 전송 실패: {response.status_code}")
        except httpx.RequestError as e:
            print(f"라벨 전송 중 오류 발생: {e}")


async def send_label_and_image_to_spring(label, frame, max_retries=3):
    spring_boot_url = "http://localhost:8080/api/send-label-and-image"

    for attempt in range(max_retries):
        try:
            # 이미지를 base64로 인코딩
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
                await asyncio.sleep(2 ** attempt)  # 지수 백오프
            continue

    print("최대 재시도 횟수 초과")
    return False


def save_image(frame, label):
    # 저장할 디렉토리 생성
    save_dir = "fall_detection_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 파일명 생성 (timestamp + label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/fall_{timestamp}_{label}.jpg"

    # 이미지 저장
    cv2.imwrite(filename, frame)
    print(f"이미지 저장됨: {filename}")
    return filename


async def generate_frames():
    global Label_List, boolHumanCheck, nNotDetected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        with torch.no_grad():
            outputs = keypoint_model(input_tensor)

        for output in outputs:
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
                                    print("--------------------")
                                else:
                                    print("위험 감지 알림 전송 성공")
                            elif 1 in Label_List and nNotDetected >= 4:
                                box_color = (0, 0, 255)
                                show_message()
                                success = await send_label_and_image_to_spring(First_Label, frame)
                                if not success:
                                    print("위험 감지 알림 전송 실패")
                                    print("--------------------")
                                else:
                                    print("위험 감지 알림 전송 성공")
                                    print("--------------------")
                            else:
                                box_color = (0, 255, 100)
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

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/")
async def index():
    return Response(content="""
    <html>
        <head>
            <title>실시간 영상</title>
        </head>
        <body>
            <h1>실시간 영상</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """, media_type="text/html")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


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
