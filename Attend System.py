import cv2
import os
import numpy as np
import pandas as pd
import tkinter as tk


from datetime import datetime
def record_attendance(name):
    global last_date, names_recorded_today, attendance_records
    
    now = datetime.now()
    current_date = now.date()
    current_time = now.strftime("%H:%M:%S")
    print(f'Welcome {name} on {current_date}')
    
    if name not in names_recorded_today:
        attendance_records.append({'Name': name, 'Date': current_date, 'Time': current_time})
        names_recorded_today.add(name)
        print(f'Attendance recorded for {name} at {current_time}')  
        

        attendance = pd.DataFrame(attendance_records)
        attendance.to_excel('attendance.xlsx', index=False)
        print("Attendance saved to Excel") 
    else:
        print(f'{name} has already been recorded today.')

    last_date = current_date

def on_button_click(event, x, y, flags, param):
    global attendance_records
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if 70 <= y <= 110:  
            record_attendance(param)
        elif 10 <= y <= 50:  
            print("Quitting...")
            attendance = pd.DataFrame(attendance_records)
            print(attendance) 
            attendance.to_excel('attendance.xlsx', index=False)
            print("Attendance saved to Excel") 
            cv2.destroyAllWindows()

path = "Training"

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images = []
labels = []
label_to_name = {}

dataset_path = "Dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

for filename in os.listdir(path):
    if filename.endswith('.jpg'):  
        img = cv2.imread(os.path.join(path, filename))
        
        if img is None:
            print(f'Failed to load image {filename}')
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (100, 100))
        
        label, name = filename.split('_') 
        label = int(label)
        name = name.split('.')[0]  
        
        label_folder = os.path.join(dataset_path, name)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        img_path = os.path.join(label_folder, f"{filename}")
        cv2.imwrite(img_path, gray_resized)

        images.append(gray_resized)
        label_to_name[label] = name
        labels.append(label)

if not images:
    print('No images were loaded')
if not labels:
    print('No labels were loaded')


recognizer.train(images, np.array(labels))


cap = cv2.VideoCapture(0)
attendance_records = []

last_date = None
names_recorded_today = set()  

while True:
    ret, img = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        
        name = label_to_name.get(label, 'Unknown')
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        
        cv2.rectangle(img, (10, 10), (140, 50), (0, 255, 0), -1) 
        cv2.putText(img, "Quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(img, (10, 70), (140, 110), (255, 0, 0), -1) 
        cv2.putText(img, "Attend", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('img', img)

        cv2.setMouseCallback('img', on_button_click, param=name)  

    key = cv2.waitKey(1)
    if key == 27:  
        print("Quitting...")
        attendance = pd.DataFrame(attendance_records)
        print(attendance)
        attendance.to_excel('attendance.xlsx', index=False)
        print("Attendance saved to Excel")
        break

cap.release()
cv2.destroyAllWindows()
