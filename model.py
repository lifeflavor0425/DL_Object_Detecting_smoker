from ultralytics import YOLO
import cv2
import numpy as np
from glob import glob
import sys
import time

# 모델 파라미터 세팅
# def setting(model) :
#   model.overrides['conf'] = 0.5  # NMS confidence threshold
#   model.overrides['iou'] = 0.5  # NMS IoU threshold
#   model.overrides['agnostic_nms'] = False  # NMS class-agnostic
#   model.overrides['max_det'] = 100  # maximum number of detections per image
#   return model

# 라이브 캠
# results = model.predict(source="0", show=True)

# 테스트 이미지 폴더
# results = model.predict(source="./test.txt", save=True) # Display preds. Accepts all YOLO predict arguments

# 이미지 확인
# from PIL
# im1 = Image.open("./smoke2.jpeg")
# results = ciga_model.predict(source='./smoke2.jpeg', save=True)  # save plotted images

# 이미지 ndarray 확인
# from ndarray
# im2 = cv2.imread("./smoker1.jpeg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# 둘다 확인
# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])

live_cam = None


def init():
    ciga_model = YOLO("./models/ciga/ciga_v8s.pt")
    smoke_model = YOLO("./models/smoke/smoke_m_adam_best.pt")
    person_model = YOLO("./models/person/person_v8s.pt")
    return ciga_model, smoke_model, person_model


def get_models(stream=True):
    ciga_model, smoke_model, person_model = init()
    # , cv2.CAP_DSHOW
    if stream:
        capture = cv2.VideoCapture(0)
        global live_cam
        live_cam = capture
        time.sleep(2)
        if not capture.isOpened():
            sys.exit("카메라 연결 오류")
    else:
        capture = cv2.VideoCapture("static/video/smoking_me.mp4")
        time.sleep(2)
        if not capture.isOpened():
            sys.exit("동영상 연결 오류")
    return capture, ciga_model, smoke_model, person_model


def cam_close():
    if live_cam.isOpened():
        live_cam.release()
