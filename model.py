from ultralytics import YOLO
import cv2
import sys
import time

# 모델 파라미터 세팅
# def setting(model) :
#   model.overrides['conf'] = 0.5  # NMS confidence threshold
#   model.overrides['iou'] = 0.5  # NMS IoU threshold
#   model.overrides['agnostic_nms'] = False  # NMS class-agnostic
#   model.overrides['max_det'] = 100  # maximum number of detections per image
#   return model

# 기본 캠 None 세팅(캠 메모리 해제용)
live_cam = None

# 모델 초기화
def init():
    ciga_model = YOLO("./models/ciga/ciga_v8s.pt")
    smoke_model = YOLO("./models/smoke/maybe_n_smoke_best.pt")
    person_model = YOLO("./models/person/person_v8s.pt")
    return ciga_model, smoke_model, person_model


# stream(캠) 인지 video인지 식별 후 영상 정보 반환
def get_models(stream=True):
    ciga_model, smoke_model, person_model = init()
    if stream:
        # 안될 때 -> cv2.CAP_DSHOW
        capture = cv2.VideoCapture(0)
        global live_cam
        live_cam = capture
        time.sleep(2)
        if not capture.isOpened():
            sys.exit("카메라 연결 오류")
    else:
        capture = cv2.VideoCapture("static/video/video1_AdobeExpress.mp4")
        time.sleep(2)
        if not capture.isOpened():
            sys.exit("동영상 연결 오류")
    return capture, ciga_model, smoke_model, person_model


# stream(캠) 메모리 해제
def cam_close():
    if live_cam.isOpened():
        live_cam.release()
