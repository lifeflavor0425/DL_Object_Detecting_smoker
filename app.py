from flask import Flask, render_template, Response
import cv2
from model import get_models, cam_close
import sys
import time
from flask_socketio import SocketIO
from glob import glob
from dlib_py import determine

app = Flask(__name__)
app.config["SECRET_KEY"] = "smoker"
socketio = SocketIO(app)


def drawing_boxes(boxes, names, frame):
    color = (255, 102, 153)
    return_conf = 0
    for box, name in zip(boxes, names):
        x1, y1, x2, y2, confidence, _ = box[0].numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if confidence < 0.4:
            return_conf = 0
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name}-{confidence}"
        cv2.putText(frame, text, (x1, y1 + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        return_conf = confidence
    return return_conf


def detect_live(model, frame):
    result = model.predict(frame)
    return result[0]


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


def get_smoker(names):
    img_name = glob("static/detected_model_images/*.jpg")[-1]
    data = {"smoker": names, "img": img_name}

    socketio.emit("get_smoker", data)


def video_stream(capture, ciga_model, smoke_model, person_model):
    init = [None] * 3
    detected_result = init
    cnt = 0
    while True:
        res, frame = capture.read()
        frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_CUBIC)

        if not res:
            sys.exit("프레임 정보 획득 오류, 여기서는 종료처리")
        else:
            flag = cnt % 15
            if flag == 0:
                detected_result = init
                detected_result[0] = detect_live(person_model, frame)
            elif flag == 5:
                detected_result[1] = detect_live(ciga_model, frame)
            elif flag == 10:
                detected_result[2] = detect_live(smoke_model, frame)
                pass

            # smoker 정보를 하나씩 뒤지면서 화면에 박스 및 검출명 드로잉
            boxes, names = boxes_and_names(detected_result)
            confidence = drawing_boxes(boxes, names, frame)
            if len(boxes) > 2 and confidence >= 0.4:
                date = time.strftime("%y-%m-%d_%X").split("-")[-1].replace(":", "_")  # 날짜 - 시간
                cv2.imwrite(f"static/detected_model_images/{date}.jpg", frame)
                get_smoker(names)

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            cnt += 1
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@socketio.on('get_recidivist')
def get_recidivist():
    recidivist_person = glob("static/recidivist_person_image/*.jpg")  # 상습범(현상수배범)
    find_recidivist = glob("static/detected_model_images/*.jpg")
    determine(find_recidivist, recidivist_person)
    result_img = glob("static/dlib_result_image/*.jpg")
    data = {"img": result_img}
    socketio.emit("get_result", data)

@socketio.on('get_pathname')
def get_pathname(data):
    path = data['path']
    if path == '/stream' :
        cam_close()
    

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ppt")
def ppt():
    return render_template("pages/ppt.html")


@app.route("/stream")
def stream():
    return render_template("pages/stream.html")


@app.route("/stream_feed")
def stream_feed():
    capture, ciga_model, smoke_model, person_model = get_models()
    return Response(
        video_stream(capture, ciga_model, smoke_model, person_model),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video")
def video():
    return render_template("pages/video.html")

@app.route('/video_feed')
def video_feed():
    capture, ciga_model, smoke_model, person_model = get_models(False)
    return Response(
        video_stream(capture, ciga_model, smoke_model, person_model),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/recidivist")
def recidivist():
    return render_template("pages/recidivist.html")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=3333)
