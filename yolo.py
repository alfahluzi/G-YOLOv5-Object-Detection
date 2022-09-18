# YOLO object detection
from statistics import mode
from unittest import result
from time import time
from PIL import Image
import cv2 as cv
import dxcam
import torch
"""
# References & Original Documentation
https://github.com/ultralytics/yolov5/issues/36
https://github.com/ra1nty/DXcam
# class names (Use index number)
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
'teddy bear', 'hair drier', 'toothbrush']
"""
def screenSize(x = 0, y= 0, width = 640, height = 640):
    left = int (1920 - (1920-x))
    top =  int (1080 - (1080-y))
    right = int (left + width)
    bottom = int (top + height)
    region = (left, top, right, bottom)
    return region

screen = dxcam.create()
region = screenSize(360, 300)
screen.start(region=region)


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.35
model.classes = [0, 2]
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

while True:
    start_time = time()
    arrimg = screen.get_latest_frame()
    arrimg = cv.cvtColor(arrimg, cv.COLOR_BGR2RGBA)
    
    # Processing Image
    img = Image.fromarray(arrimg)
    result = model(img)

    # Draw Rectangle
    resultDetail = result.pandas().xyxy[0]
    color = (0,250,0)
    for i in range(len(resultDetail.name)):
        conf = resultDetail.confidence[i]
        xyMin = (int(resultDetail.xmin[i]), int(resultDetail.ymin[i]))
        xyMax = (int(resultDetail.xmax[i]), int(resultDetail.ymax[i]))
        cv.putText(arrimg, f"{resultDetail.name[i]} ({conf:.2f})", xyMin, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.rectangle(arrimg, xyMin, xyMax, color, 2)
    
    # Count FPS
    now_time = time()
    fps = 1.0/(now_time - start_time)
    print(f"Frames Per Second : {fps:.2f}")
    cv.putText(arrimg, f'FPS: {fps:.2f}', (20,70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    #Show Result
    cv.imshow("Screen", arrimg)
    
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break