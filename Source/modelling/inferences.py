
# Ignore warnings
import warnings
warnings.filterwarnings("ignore") # Warning will make operation confuse!!!

# Model
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =  torch.hub.load('ultralytics/yolov5', 'custom','./best.pt').to(device)


model.conf = 0.25 # NMS confidence threshold
model.iou = 0.45  # IoU threshold
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [3]   # (optional list) filter by class, 77 for "tendy bear"
# I suggest to see the label list in https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

# RUN
import cv2
"""video = cv2.VideoCapture(0) # Read USB Camera
while(video.isOpened()):
    # Read Frame
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    # Object Detection
    results = model(frame)
    cv2.imshow('Object detector', results.render()[0])
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'): break"""

path = './0ed19ad4-1636062728172_Isuzu_Elf_2016_Lorry.jpg'
  
# Reading an image in default mode
image = cv2.imread(path)

  
# Window name in which image is displayed
window_name = 'image'
results = model(image) 
results.show()
# Using cv2.imshow() method 
# Displaying the image

'''image = cv2.resize(image, (1280, 720)) 
cv2.imshow(window_name, image)'''
cv2.waitKey(0)


# Clean up
#video.release()
cv2.destroyAllWindows()