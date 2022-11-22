import cv2
import numpy as np 
import torch

cap = cv2 .VideoCapture(2,cv2.CAP_DSHOW)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt').to(device)
classes = ['roda_2', 'kendaraan_berat', 'roda_4']

with torch.no_grad(): 
    results = model(classes)
    results.print()
    results = results.xyxy

for result in results[0]:
    result = result.detach().numpy()
    cv2.rectangle(classes, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (255, 0, 0), 1)
    cv2.putText(classes, f"{classes[int(result[5])]}: {result[4]:.2f}", (int(result[0]), int(result[1]) - 5),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    #cv2.putText(image, f"{result[5]}")

image = cv2.resize(classes, (1280, 720))
cv2.imshow("image", image)
cv2.waitKey(0)

# Results
print(results.xyxy)  # or .show(), .save(), .crop(), .pandas(), etc.
