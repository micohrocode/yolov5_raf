import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image

#Model
model = torch.hub.load('ultralytics/yolov5','custom', path=r'C:\Users\mrosa\Documents\repos\yolov5\runs\train\results_v32\weights\best.pt')  # local repo
# Images
img = cv2.imread(r'C:\Users\mrosa\Documents\repos\yolov5\RAF-v1-3\train\images\Image_1255_jpg.rf.a19e93c6985613cc3e936bc9b58881c3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Inference
results = model(img, size=328)  # includes NMS

# Results
results.print()  
#results.show()  # or .show()

boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
predictions = {
    "pred_boxes": [],
    "pred_classes": [],
    "scores": []
}

for index, row in boxes.iterrows():
    # Access each row's data
    x1, y1, x2, y2, confidence, class_id, name = row
    # print(f"Box {index}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={confidence}, class_id={class_id}, name={name}")
    predictions['pred_boxes'].append((x1,y1,x2,y2))
    if class_id not in predictions['pred_classes']:
        predictions['pred_classes'].append(class_id)
    predictions['scores'].append(confidence)

print(predictions)

