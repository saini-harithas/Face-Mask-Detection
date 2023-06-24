from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def mask_prediction(frame, face_net, mask_net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (226, 226), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_X, start_Y, end_X, end_Y = box.astype("int")

            start_X, start_Y = (max(0, start_X), max(0, start_Y))
            end_X, end_Y = (min(w - 1, end_X), min(h - 1, end_Y))

            face = frame[start_Y:end_Y, start_X:end_X]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (226, 226))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((start_X, start_Y, end_X, end_Y))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    return locs, preds

proto_txt = r"face_detector\deploy.prototxt"
weights = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNet(proto_txt, weights)

mask_net = load_model("face_mask_detector.model")

print("\n[STATUS] Starting Video Stream.....!!!")
video = VideoStream(src=0).start()

while True:
    frame = video.read()
    frame = imutils.resize(frame, width=800)

    locs, preds = mask_prediction(frame, face_net, mask_net)

    for (box, pred) in zip(locs, preds):
        start_X, start_Y, end_X, end_Y = box
        mask, withoutMask = pred

        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.putText(frame, label, (start_X, start_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.80, color, 2)
        cv2.rectangle(frame, (start_X, start_Y), (end_X, end_Y), color, 1)

    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("x"):
        break

cv2.destroyAllWindows()
video.stop()