import cv2


video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # ลดการกระตุก

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["Unknown", "phu", "aom", "art", "F", "Mark", "Max", "Ping", "Pon", "Sin", "Tiger"]  # เปลี่ยนค่าเริ่มต้นเป็น "Unknown"
threshold = 80  

while True:
    ret, frame = video.read()
    if not ret:
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        
        
        if conf < threshold and 0 <= serial < len(name_list):
            name = name_list[serial]
            color = (0, 255, 0)  
        else:
            name = "Unknown"  
            color = (0, 0, 255)  
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frame = cv2.resize(frame, (320, 240))  
    cv2.imshow("Frame", frame)

    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
