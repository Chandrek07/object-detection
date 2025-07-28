import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import *

model = YOLO(r'C:\Users\Lenovo\Desktop\object detection\yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(r'C:\Users\Lenovo\Desktop\object detection\veh2\veh2.mp4')

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0
tracker = Tracker()
cy1, cy2 = 322, 368
offset = 6
entry_count = 0
exit_count = 0

# Line positions
line_up = (274, cy1), (814, cy1)
line_down = (177, cy2), (927, cy2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    detect_list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        conf = row[4]
        cls_id = int(row[5])
        cls_name = class_list[cls_id]
        if 'car' in cls_name:
            detect_list.append([x1, y1, x2, y2])
            cv2.putText(frame, f'{cls_name} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    bbox_id = tracker.update(detect_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(obj_id), (cx, cy),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy1 - offset < cy < cy1 + offset:
            entry_count += 1

        elif cy2 - offset < cy < cy2 + offset:
            exit_count += 1

    # Draw lines
    cv2.line(frame, *line_up, (255, 255, 255), 2)
    cv2.line(frame, *line_down, (255, 255, 255), 2)

    # Display count
    cv2.putText(frame, f'Entries: {entry_count}', (50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Exits: {exit_count}', (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite(f'frame_{count}.jpg', frame)
    elif key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()
