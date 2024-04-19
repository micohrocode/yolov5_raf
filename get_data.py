from roboflow import Roboflow
rf = Roboflow(api_key="HKMk0L6CtU7cgT4DytZt")
project = rf.workspace("raf-uovpx").project("raf-v1")
version = project.version(3)
dataset = version.download("yolov5")


# py train.py --img 416 --batch 16 --epochs 50 --data RAF-v1-3/data.yaml --cfg ./models/yolov5s.yaml --weights 'yolov5s.pt' --name results_v3

# py detect.py --weights runs/train/results_v3/weights/best.pt --img 416 --conf 0.4 --source RAF-v1-3/test/images

# yolov5\RAF-v1-3\train\images\Image_407_png.rf.8bc604a5e0b12751e910df4517a8e35a.jpg