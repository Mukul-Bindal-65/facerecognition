import cv2
import face_recognition
cap=cv2.VideoCapture(0)
import os
try:
    os.mkdir('facedataset')
except:
    print("Already directory exist, no need to create")

i=1

face_name='Mukul_Bindal'

try:
    os.mkdir(f'facedataset/{face_name}')
except:
    print("Already directory exist, no need to create")


while True:
    ret,frame=cap.read()
    cv2.imshow("Camera",frame)
    cv2.imwrite(f'facedataset/{face_name}/{face_name}_{i}.png',frame)
    i+=1
    cv2.waitKey(600)
    if i==32:
        break

cap.release()
cv2.destroyAllWindows()