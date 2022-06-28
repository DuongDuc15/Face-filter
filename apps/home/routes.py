# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, Response, jsonify
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound

# ---------------------------------------
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from .apply_mask import applyMask
from os import listdir
from os.path import isfile, join
from itertools import cycle

global capture, rec_frame, listmask, len_list, switch, face, rec, out
capture = 0
face = 0
switch = 1
rec = 0
listmask = [f for f in listdir("./mask") if isfile(join("./mask", f))]
index_list = 0


# model load
modelFile = r"models\dnn\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"models\dnn\deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# landmarks load
facemark = cv2.face.createFacemarkLBF()
facemark_path = r"models\lbf\lbfmodel.yaml"
facemark.loadModel(facemark_path)

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(r"D:\Project\AIP\data\Mark Ronson  Uptown Funk Official Video ft Bruno Mars_480p.mp4")

folder_path = ""
rec_frame = None


def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame, mask):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    ########## detect 1 face
    # confidence = detections[0, 0, 0, 2]

    # if confidence < 0.5:
    #         return frame
    # box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])

    # (startX, startY, endX, endY) = box.astype("int")
    # (x, y, x1, y1) = box.astype("int")

    # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

    ########## detect multiple faces
    face_mark = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.90:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            face_mark.append([x, y, x1 - x, y1 - y])

    landmarks = None
    if len(face_mark) > 0:
        try:
            face_mark = np.array(face_mark)
            status, landmarks = facemark.fit(frame, face_mark)

            nose_image = cv2.imread(
                os.path.sep.join(
                    [
                        "./mask",
                        mask,
                    ]
                )
            )

            frame = applyMask(frame, landmarks, nose_image)
        except:
            pass

    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            mask = listmask[index_list]
            if mask:
                frame = detect_face(frame, mask)
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    [
                        folder_path + "/images",
                        "shot_{}.png".format(
                            str(now).replace(":", "").replace(" ", "")
                        ),
                    ]
                )
                print(p)
                cv2.imwrite(p, cv2.flip(frame, 1))

            if rec:
                rec_frame = frame
                frame = cv2.putText(
                    cv2.flip(frame, 1),
                    "Recording...",
                    (0, 25),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode(".jpg", cv2.flip(frame, 1))

                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            except Exception as e:
                pass

        else:
            pass


@blueprint.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@blueprint.route("/requests", methods=["POST", "GET"])
def tasks():
    global switch, camera, folder_path
    folder_path = current_user.folder_path

    if request.method == "POST":
        if request.form.get("click") == "Capture":
            global capture
            capture = 1
        elif request.form.get("face") == "Change":
            global index_list
            if index_list == len(listmask) - 1:
                index_list = 0
            else:
                index_list = index_list + 1
        elif request.form.get("stop") == "Stop/Start":

            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get("rec") == "Start/Stop Recording":
            global rec, out
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                p = os.path.sep.join(
                    [
                        folder_path + "/videos",
                        "vid_{}.avi".format(str(now).replace(":", "")),
                    ]
                )
                out = cv2.VideoWriter(p, fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(
                    target=record,
                    args=[
                        out,
                    ],
                )
                thread.start()
            elif rec == False:
                out.release()

    elif request.method == "GET":
        return render_template("home/index.html")
    return render_template("home/index.html")


@blueprint.route("/index", methods=["GET", "POST"])
@login_required
def index():
    global folder_path
    folder_path = current_user.folder_path
    return render_template("home/index.html", segment="index")


@blueprint.route("/table")
@login_required
def table():
    global folder_path
    folder_path = current_user.folder_path
    video_path = folder_path + "/videos"
    image_path = folder_path + "/images"
    video_list = image_list = []
    video_list = [
        "/shots/username_" + current_user.username + "/videos/" + f
        for f in listdir(video_path)
        if isfile(join(video_path, f))
    ]
    image_list = [
        "/shots/username_" + current_user.username + "/images/" + f
        for f in listdir(image_path)
        if isfile(join(image_path, f))
    ]

    return render_template(
        "/home/table-basic.html",
        segment="table",
        video_list=video_list,
        video_len=len(video_list),
        image_list=image_list,
        image_len=len(image_list),
    )


@blueprint.route("/<template>")
@login_required
def route_template(template):

    try:

        if not template.endswith(".html"):
            template += ".html"

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template("home/page-404.html"), 404

    except:
        return render_template("home/page-500.html"), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split("/")[-1]

        if segment == "":
            segment = "index"

        return segment

    except:
        return None
