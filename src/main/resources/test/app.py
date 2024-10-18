import cv2
import numpy as np
import onnxruntime
import math
from flask import Flask, render_template, Response

app = Flask(__name__)

# ONNX 모델 로드
onnx_model_path = 'keypointrcnn_cpu.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# 모델의 입력 이름과 형태 출력
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print("예상되는 입력 형태:", input_shape)

def preprocess_image(image):
    height, width = image.shape[:2]
    new_height = input_shape[2]  # 224
    new_width = input_shape[3]   # 224

    # 원본 이미지와 새로운 크기의 비율 계산
    scale = min(new_width / width, new_height / height)

    # 이미지의 새로운 크기 계산
    resized_width = int(width * scale)
    resized_height = int(height * scale)

    # 이미지 리사이즈
    image_resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # 패딩 추가하여 224x224 맞추기
    top_pad = (new_height - resized_height) // 2
    bottom_pad = new_height - resized_height - top_pad
    left_pad = (new_width - resized_width) // 2
    right_pad = new_width - resized_width - left_pad

    image_padded = cv2.copyMakeBorder(image_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 정규화
    image_normalized = image_padded.astype(np.float32) / 255.0
    input_data = np.expand_dims(image_normalized, axis=0).transpose(0, 3, 1, 2)
    return input_data, scale, (left_pad, top_pad)

def calculate_angle(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    return math.degrees(math.atan2(delta_y, delta_x))

# 키포인트들을 연결하여 스켈레톤을 그리는 함수
def draw_skeleton(image, keypoints, skeleton_pairs, scale, padding, threshold=0.5):
    left_pad, top_pad = padding
    for i, kp in enumerate(keypoints):
        if kp[2] > threshold:
            x = int((kp[0] - left_pad) / scale)
            y = int((kp[1] - top_pad) / scale)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(image, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    for pair in skeleton_pairs:
        p1, p2 = keypoints[pair[0]], keypoints[pair[1]]
        if p1[2] > threshold and p2[2] > threshold:
            p1_x, p1_y = int((p1[0] - left_pad) / scale), int((p1[1] - top_pad) / scale)
            p2_x, p2_y = int((p2[0] - left_pad) / scale), int((p2[1] - top_pad) / scale)
            cv2.line(image, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 0), 1)

# 수정된 스켈레톤 연결 쌍
skeleton_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame, (320, 240))  # 프레임 크기 축소
            input_data, scale, padding = preprocess_image(frame)
            predictions = session.run(None, {input_name: input_data})
            keypoints = predictions[3][0]
            draw_skeleton(frame, keypoints, skeleton_pairs, scale, padding)
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
