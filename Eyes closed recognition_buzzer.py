import cv2
import time
import RPi.GPIO as GPIO

# GPIO 18핀 부저 연결
GPIO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPIO_PIN, GPIO.OUT)

# OpenCV 설정
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 눈 감지 함수 예제
def detect_eyes(gray, frame):
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return len(eyes) > 0

def main():
    cap = cv2.VideoCapture(0)  # 카메라를 사용하여 비디오 스트림 열기

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

        eyes_detected = detect_eyes(gray, frame)

        if eyes_detected:
            start_time = time.time()  # 눈이 감겨 있는지 확인하기 위한 시작 시간 기록
        else:
            if time.time() - start_time > 5:  # 약 4초동안 눈이 일정 시간 감겨 있으면
                GPIO.output(GPIO_PIN, GPIO.HIGH)  # 버저 울리기
            else:
                GPIO.output(GPIO_PIN, GPIO.LOW)  # 버저 끄기

        cv2.imshow('Eye Detection', frame)  # 화면에 프레임 표시

        if cv2.waitKey(1) & 0xFF == ord('e'):  # 'e' 키를 누르면 openCV 종료
            break

    cap.release()
    cv2.destroyAllWindows()

    GPIO.cleanup()  # GPIO 설정 초기화

if __name__ == "__main__":
    main()