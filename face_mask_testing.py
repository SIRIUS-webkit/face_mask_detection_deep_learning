import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

print("[INFO] loading model...")

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("testv2.mp4")
cap.set(3, 720)
cap.set(4, 580)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
model = load_model('face_mask_detector_modelv1.h5')
classes = ['with_mask', 'without_mask']
print("[INFO] starting video stream...")

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()


    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        roi_img = frame[startY:endY, startX:endX]

        roi_img = cv2.resize(roi_img, (112,112), interpolation=cv2.INTER_CUBIC)

        roi_img = roi_img.astype("float") / 255.0
        roi_img = img_to_array(roi_img)
        roi_img = np.expand_dims(roi_img, axis=0)
        result = model.predict(roi_img)[0]
        idx = np.argmax(result)
        print(idx)
        label = classes[idx]

        cv2.putText(frame, label, (startX, startY - 30), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFf == ord('q'):
        break


cv2.destroyAllWindows()
