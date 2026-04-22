import cv2
import glob
import os

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID: ")

existing_files = glob.glob(f"datasets/User.{id}.*.jpg")

count = len(existing_files)  
print(f"Starting from {count + 1}...")

video = cv2.VideoCapture(1)  

while True:
    ret, frame = video.read()
    if not ret:
        print("ไม่สามารถเข้าถึงกล้องได้")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        filename = f'datasets/User.{id}.{count}.jpg'
        cv2.imwrite(filename, gray[y:y+h, x:x+w])  

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or (count - len(existing_files)) >= 200:
        break

video.release()
cv2.destroyAllWindows()
print(f"Dataset Collection Done. Total images: {count}")

