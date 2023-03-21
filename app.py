from flask import Flask, render_template, Response
import cv2
from model import get_models, cam_close
import sys
import time
from flask_socketio import SocketIO
from glob import glob
from dlib_py import determine

# Flask & socket 세팅
app = Flask(__name__)
app.config["SECRET_KEY"] = "smoker"
socketio = SocketIO(app)

# frame에 boxing
def drawing_boxes(boxes, names, frame):
    color = (255, 102, 153)
    return_conf = 0
    for box, name in zip(boxes, names):
        x1, y1, x2, y2, confidence, _ = box[0].numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name}-{confidence:.4f}"
        cv2.putText(frame, text, (x1, y1 + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return_conf = confidence
    return return_conf


# model 예측 결과 반환
def detect_live(model, frame):
    result = model.predict(frame)
    return result[0]


# 각 모델의 예측 결과(boxing값, class값)들을 리스트에 담아서 반환
def boxes_and_names(detected_result):
    boxes = list()
    names = list()
    for value in detected_result:
        if value is None:
            continue
        if not value.boxes.numpy():
            continue
        boxes.append(value.boxes.data)
        names.append(value.names[0])

    return boxes, names


# socket emit -> 흡연자 이미지와 class값 전달
def get_smoker(names):
    img_path = glob("static/detected_model_images/*.jpg")[-1]
    data = {"smoker": names, "img": img_path}

    socketio.emit("get_smoker", data)


# 영상의 각 frame을 yelid로 하나씩 전달
def video_stream(capture, ciga_model, smoke_model, person_model):
    init = [None] * 3
    detected_result = init
    cnt = 0
    while True:
        res, frame = capture.read()
        if not res:
            sys.exit("프레임 정보 획득 오류, 여기서는 종료처리")
        else:
            flag = cnt % 60
            if flag == 0:
                detected_result[0] = detect_live(person_model, frame)
            elif flag == 20:
                detected_result[1] = detect_live(smoke_model, frame)
            elif flag == 40:
                detected_result[2] = detect_live(ciga_model, frame)

            # smoker 정보를 하나씩 뒤지면서 화면에 박스 및 검출명 드로잉
            boxes, names = boxes_and_names(detected_result)
            confidence = drawing_boxes(boxes, names, frame)
            if len(boxes) > 2 and confidence >= 0.4:
                date = time.strftime("%y-%m-%d_%X").split("-")[-1].replace(":", "_")
                cv2.imwrite(f"static/detected_model_images/{date}.jpg", frame)
                get_smoker(names)

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            cnt += 1
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# dlib -> 상습 흡연자 적발
@socketio.on("get_recidivist")
def get_recidivist():
    recidivist_person = glob("static/recidivist_person_image/*.jpg")  # 상습범(현상수배범)
    find_recidivist = glob("static/detected_model_images/*.jpg")
    determine(find_recidivist, recidivist_person)
    result_img = glob("static/dlib_result_image/*.jpg")
    data = {"img": result_img}
    socketio.emit("get_result", data)


# opencv 캠 메모리 해제
@socketio.on("get_pathname")
def get_pathname(data):
    path = data["path"]
    if path == "/stream":
        cam_close()


# Home(index) 페이지
@app.route("/")
def index():
    return render_template("index.html")


# PPT 발표용 페이지
@app.route("/ppt")
def ppt():
    return render_template("pages/ppt.html")


# 라이브 캠 페이지
@app.route("/stream")
def stream():
    return render_template("pages/stream.html")


# 라이브 캠 페이지에 frame 전달
@app.route("/stream_feed")
def stream_feed():
    capture, ciga_model, smoke_model, person_model = get_models()
    return Response(
        video_stream(capture, ciga_model, smoke_model, person_model),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# video 페이지
@app.route("/video")
def video():
    return render_template("pages/video.html")


# video 페이지에 frame 전달
@app.route("/video_feed")
def video_feed():
    capture, ciga_model, smoke_model, person_model = get_models(False)
    return Response(
        video_stream(capture, ciga_model, smoke_model, person_model),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# recidivist(상습범) 페이지
@app.route("/recidivist")
def recidivist():
    return render_template("pages/recidivist.html")


# 실행
if __name__ == "__main__":
    socketio.run(app, debug=True, port=3333)
