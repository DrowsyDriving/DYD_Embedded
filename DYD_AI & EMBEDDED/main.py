import cv2
import dlib
import time
import numpy as np
import requests
from scipy.spatial import distance

# 모델 로드
detect = './detect/'
prototxt = detect + 'deploy.prototxt'
resnet = detect + 'res10_300x300_ssd_iter_140000.caffemodel'
face_model = cv2.dnn.readNet(resnet, prototxt)

landmark = detect + 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmark)

# 카메라 로드 (웹캠)
cam = cv2.VideoCapture(0)

# 최소 인식률
minimum_confidence = 0.5

# 시간 측정 변수
check_time = None
closed_eye = False
total = 0
warning_count = 0

# 플라스크 서버 URL
server_url = 'http://your_server_ip:5000/warning'

# 눈 감은 정도 확인 함수
def ear(eyes):
    a = distance.euclidean(eyes[1], eyes[5])
    b = distance.euclidean(eyes[2], eyes[4])
    c = distance.euclidean(eyes[0], eyes[3])
    eye_aspect_ratio = (a + b) / (2.0 * c)
    return eye_aspect_ratio

# 서버로 데이터 전송 함수
def send_data(car_number, warning_level, latitude, longitude):
    data = {
        'car_number': car_number,
        'warning_level': warning_level,
        'latitude': latitude,
        'longitude': longitude
    }
    response = requests.post(server_url, json=data)
    if response.status_code == 200:
        print("Data sent successfully!")
    else:
        print("Failed to send data to server.")

# 위치 정보 가져오는 함수
def get_location():
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    return data

while True:
    ret, frame = cam.read()

    if not ret:
        print('not ret')
        break

    # 얼굴 먼저 인식
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 124.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    faces = []

    for i in range(0, detections.shape[2]):
        # 얼굴 인식 확률 추출
        confidence = detections[0, 0, i, 2]
        if confidence > minimum_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype('int')
            faces.append(dlib.rectangle(start_x, start_y, end_x, end_y))

    # 얼굴 인식 후 랜드마크로 눈 인식
    for face in faces:
        face_landmark = predictor(frame, face)

        left_eyes = []
        right_eyes = []
        for eye in range(36, 42):  # parts : 전체 구하기 / part(n) : n 부분 구하기
            left_eyes.append([face_landmark.part(eye).x, face_landmark.part(eye).y])
            right_eyes.append([face_landmark.part(eye + 6).x, face_landmark.part(eye + 6).y])

        # 눈 감은 정도를 이용해서 시간을 측정
        if closed_eye:
            total += time.time() - check_time
            if total >= 5.0:
                location_data = get_location()
                latitude = location_data.get('loc').split(',')[0]
                longitude = location_data.get('loc').split(',')[1]
                send_data(car_number='car_number', warning_level=warning_count, latitude=latitude, longitude=longitude)
                warning_count += 1
                total = 0
        else:
            total = 0

        left_eye, right_eye = ear(left_eyes), ear(right_eyes)
        if left_eye < 0.15 and right_eye < 0.15:
            closed_eye = True
            check_time = time.time()
        else:
            closed_eye = False

    cv2.imshow('test', frame)

    if cv2.waitKey(33) == 27:
        break

cam.release()
cv2.destroyAllWindows()
