import cv2
import face_recognition
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


# face_recognition.load_image_file(file)


# here im loading my web
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    #detect faces 
    face_locations,face_names = sfr.detect_known_faces(frame)
    for face_loc , name in zip(face_locations, face_names):
        y1,x2,y2,x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,1, (42,202,10),2)
        
        cv2.rectangle(frame,(x1,y1),(x2,y2),(42,203,10),4)
    
    cv2.imshow("Frame",frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

