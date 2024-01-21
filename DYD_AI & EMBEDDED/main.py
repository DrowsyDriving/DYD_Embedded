import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance
import requests
from geopy.geocoders import Nominatim
import RPi.GPIO as GPIO

# GPIO 핀 번호 설정
BUZZER_PIN = 18

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

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

# Flask 서버 엔드포인트
flask_server_endpoint = '서버 주소 입력'

# 번호판 및 위치 정보를 저장할 변수
license_plate = "123가4567"  # 여기에 원하는 번호판을 입력하세요

# 눈 감은 정도 확인 함수
def ear(eyes):
    a = distance.euclidean(eyes[1], eyes[5])
    b = distance.euclidean(eyes[2], eyes[4])
    c = distance.euclidean(eyes[0], eyes[3])
    eye_aspect_ratio = (a + b) / (2.0 * c)
    return eye_aspect_ratio

# 위치 정보 가져오기 위한 geopy 설정
geolocator = Nominatim(user_agent="eye_tracking_app")

# 시간 측정 변수
check_time = None
closed_eye = False

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

            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(w - 1, end_x), min(h - 1, end_y)

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
                # 위치 정보 가져오기
                location = geolocator.reverse((h, w), language='en')  # 현재 좌표를 주소로 변환
                location_info = location.address if location else "Unknown Location"

                # 데이터를 Flask 서버로 전송
                data = {
                    'license_plate': license_plate,
                    'location': location_info
                }

                try:
                    response = requests.post(flask_server_endpoint, json=data)
                    print(response.text)

                    # 눈을 5초 이상 감은 경우 버저 울리기
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(1)  # 버저 울림 시간 설정 (1초)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)

                except requests.exceptions.RequestException as e:
                    print(f"서버에 데이터 전송 중 오류 발생: {e}")

                closed_eye = False  # 한 번만 전송 후 눈 감음 상태 초기화

        else:
            total = 0

        # 눈 감은 상태 갱신
        left_eye, right_eye = ear(left_eyes), ear(right_eyes)
        if left_eye < 0.15 and right_eye < 0.15:
            closed_eye = True
            check_time = time.time()
        else:
            closed_eye = False

    cv2.imshow('test', frame)

    if cv2.waitKey(33) == 27:
        break

# 루프 종료 시 GPIO 리소스 해제 및 OpenCV 창 닫기
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()
