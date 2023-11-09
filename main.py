from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

video = cv2.VideoCapture(0)
model = YOLO('model/yolov8n.pt')
results = []

def getYolo(results):
    return results.tojson()

def gen(video):
    global results
    while True:
        success, image = video.read()
        results = model(image)
        annotated_frame = results[0].plot()

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        frame = jpeg.tobytes()
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route("/")
def report():
    return render_template("report.html")

@app.route("/keluar")
def keluar():
    return render_template("keluar.html")

@app.route("/video")
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/yolo")
def write_yolo():
    return jsonify(getYolo(results[0]))

@app.route("/stop", methods=["POST", "GET"])
def stop():
    video.release()
    cv2.destroyAllWindows()
    return render_template("report.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)