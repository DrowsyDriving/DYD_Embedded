import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance
import RPi.GPIO as GPIO  # GPIO 라이브러리 추가

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

# GPIO 설정
BUZZER_PIN = 18  # 실제로 연결된 GPIO 핀으로 대체
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# 시간 측정 변수
check_time = None
closed_eye = False
total = 0

# 눈 감은 정도 확인 함수
def ear(eyes):
    a = distance.euclidean(eyes[1], eyes[5])
    b = distance.euclidean(eyes[2], eyes[4])
    c = distance.euclidean(eyes[0], eyes[3])
    eye_aspect_ratio = (a + b) / (2.0 * c)
    return eye_aspect_ratio

# 버저 제어 함수
def buzzer_alert():
    for _ in range(3):  # 3번 띠 띠 띠 소리 울리기
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.5)

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
        for eye in range(36, 42):  
            left_eyes.append([face_landmark.part(eye).x, face_landmark.part(eye).y])
            right_eyes.append([face_landmark.part(eye + 6).x, face_landmark.part(eye + 6).y])

        # 눈 감은 정도를 이용해서 시간을 측정
        if closed_eye:
            total += time.time() - check_time
            if total >= 5.0:  # 5초 이상 눈을 감으면
                buzzer_alert()  # 3번 띠 띠 띠 소리 울리기
                closed_eye = False  # 눈을 다시 감은 상태로 갱신
        else:
            total = 0
            GPIO.output(BUZZER_PIN, GPIO.LOW)  # 버저 끄기

        left_eye, right_eye = ear(left_eyes), ear(right_eyes)
        if left_eye < 0.15 and right_eye < 0.15:
            closed_eye = True
            check_time = time.time()

    cv2.imshow('test', frame)

    if cv2.waitKey(33) == 27:
        break

# GPIO 리소스 해제
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()
