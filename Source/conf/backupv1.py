import cv2 
import numpy as np 
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt').to(device)
classes = ['roda_2', 'kendaraan_berat', 'roda_4']
colors = list(np.random.rand(80,3)*255) 


cap = cv2.VideoCapture(0) 
if cap.isOpened(): 
        window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE) 
        # Window 
        while cv2.getWindowProperty("Camera", 0) >= 0: 
                ret, frame = cap.read() 
                if ret: 
                        # detection process 
                        objs = model.detach(frame) 
                        objs.print()

                        # plotting 
                        for obj in objs: 
                                # print(obj) 
                                label = obj['label'] 
                                score = obj['score'] 
                                [(xmin,ymin),(xmax,ymax)] = obj['bbox'] 
                                color = colors[obj.index(label)] 
                                frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
                                frame = cv2.putText(frame, f"{str(label)} ({str(score)})", (xmin,ymin),
                 cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA) 

                cv2.imshow("Camera", frame) 
                keyCode = cv2.waitKey(0) 
                if keyCode == ord('q'): 
                        break 
        cap.release() 
        cv2.destroyAllWindows() 
