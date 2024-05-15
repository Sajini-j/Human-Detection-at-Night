import cv2
import cvzone
import math
from ultralytics import YOLO
from PIL import Image


img = cv2.imread('new_dataset/train/images/186.png')
model = YOLO('yolov8x.pt')

classNames = ["person"]


results = model(img)

# Check if results is a list
if isinstance(results, list):
    for r in results:
        for box in r.boxes:
            try:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Draw light red color bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 150), 2)

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = box.cls[0]
                name = classNames[int(cls)]
                print(f'Detected class index: {cls}, Class name: {name}')

                cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0, x1), max(35, y1)), scale=0.5)
            except IndexError as e:
                print(f'IndexError: {e}')

else:
    for box in results.xyxy[0]:
        try:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Draw light red color bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 150), 2)

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = box.cls[0]
            name = classNames[int(cls)]
            print(f'Detected class index: {cls}, Class name: {name}')

            cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0, x1), max(35, y1)), scale=0.5)
        except IndexError as e:
            print(f'IndexError: {e}')

# Display the image with light red color bounding boxes
img1=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()
    