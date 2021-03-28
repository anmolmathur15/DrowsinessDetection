import numpy as np
import cv2
from keras.models import load_model

model = load_model("Drowsy.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

video = cv2.VideoCapture(0)

while(True):
    ret, img = video.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        flg = False
        for (ex,ey,ew,eh) in eyes:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            eyeimg = gray[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(eyeimg, (90, 90))
            lastimg = np.array(lastimg)
            lastimg = np.expand_dims(lastimg, 0)
            pred = np.argmax(model.predict(lastimg),axis=-1)
            # print(model.predict(lastimg))
            # print(pred)
            if pred == 0 : flg = True
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (x-1, y-1) 
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        
        if flg or len(eyes)==0: 
            cv2.putText(img, 'Closed', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        else : 
            cv2.putText(img, 'Open', org, font, fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow("Camera",img)
    k = cv2.waitKey(1) & 0xff
    if k=='q':
        break

video.release()
cv2.destroyAllWindows()