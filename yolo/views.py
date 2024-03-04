from django.shortcuts import render

from django.shortcuts import render
import cv2
import numpy as np

import cv2
import numpy as np
from django.http import StreamingHttpResponse

import cv2
import numpy as np
from django.http import StreamingHttpResponse

def video_capture():
    net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
    classes = []
    with open("dnn_model/classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.4, fy=1.4)
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_classes = set()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    label = str(classes[class_id])
                    if label in ['bottle', 'person', 'cell phone']:
                        detected_classes.add(label)

        for i in range(len(outs)):
            for detection in outs[i]:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    label = str(classes[class_id])
                    color = colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if not any(item in detected_classes for item in ['bottle']):
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), -1)
            cv2.putText(frame, 'Out of Stock', (int(width/2) - 50, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()




def video_feed(request):
    # Start video capture
    video_stream = video_capture()

    # Send WhatsApp message
    # send_report_via_sms()

    # Return streaming HTTP response
    return StreamingHttpResponse(video_stream, content_type='multipart/x-mixed-replace; boundary=frame')

import os
from twilio.rest import Client




def send_report_via_sms():
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
                                    body=f'Hello, Aryan \nProduct Name : Bisleri Bottle is missing \nShelf Location : D5 shelf \nPlease refill and find the real-time video below :',

    to='whatsapp:+919653484071'
    )

    print(message.sid)

