from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * 

cap = cv2.VideoCapture(r"istockphoto-1177984994-640_adpp_is.mp4")
model = YOLO("detectionimage/yolov8n.pt")
classNames = ["person"]
mask = cv2.imread("mask.png")
tracker = Sort(max_age= 20, min_hits=3, iou_threshold=0.3)
limitsup = [220,100,384,330]
limitsdown = [200,200,384,500]

totalcountsup=[]
totalcountsdown=[]

while True:
    success ,img = cap.read()
    imagegraphics =cv2.imread(r"graphics-1.png",cv2.IMREAD_UNCHANGED)
    imgRegion = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, imagegraphics,(730,260))
    rersult = model(img, stream=True)
    detections = np.empty((0, 5))
    
    for r in rersult:
        boxes = r.boxes
        
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int (x1),int (y1), int (x2),int (y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img,(x1, y1, w, h),l=9)
            conf = math.ceil((box.conf[0]*100))/100            
            cls =int(box.cls[0])
            currentclass = classNames[cls]
            
            if currentclass == "person" and conf > 0.3:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))
                
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255), 5)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255), 5)

    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2 = int (x1),int (y1), int (x2),int (y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f"{int(id)}",(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=10)
        cx, cy = x1 +w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
        if limitsup[0]<cx< limitsup[2] and limitsup[1] - 15 <cy <limitsup[1] + 15:
            if totalcountsup.count(id) == 0:
                totalcountsup.append(id)
                cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,255,0), 5)

        
        if limitsdown[0]<cx< limitsdown[2] and limitsdown[1] - 15 <cy <limitsdown[1] + 15:
            if totalcountsdown.count(id) == 0:
                totalcountsdown.append(id)
                cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,255,0), 5)

    cv2.putText(img, str(len(totalcountsup)),(929,345),cv2.FONT_HERSHEY_PLAIN, 5, (139,195,75), 7)
    cv2.putText(img, str(len(totalcountsup)),(1191,345),cv2.FONT_HERSHEY_PLAIN, 5, (50,50,230), 7)

    cv2.imshow("ImageRegion",imgRegion)    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    
    
    