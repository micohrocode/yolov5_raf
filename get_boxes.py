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

#results = results.xyxy[0]  # img1 predictions (tensor)
boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
print(boxes)
