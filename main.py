import cv2
import time
import threading
import csv
import os
import tkinter as tk
from PIL import Image, ImageTk

def detect_faces(frame, faceNet):
    frameHeight, frameWidth, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faces = []

    for i in range(0, detection.shape[2], 2):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faces.append((x1, y1, x2, y2))

    return faces

def process_face(genderNet, ageNet, frame, face, csv_writer, age_gender):
    x1, y1, x2, y2 = face
    face_region = frame[y1:y2, x1:x2]

    # Draw a green rectangle around the face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess face for gender prediction
    gender_blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict gender
    genderNet.setInput(gender_blob)
    genderPred = genderNet.forward()
    gender = 'Male' if genderPred[0][0] > 0.5 else 'Female'

    # Preprocess face for age prediction
    age_blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Predict age
    ageNet.setInput(age_blob)
    agePred = ageNet.forward()
    ageList = ['(0-12)', '(13-17)', '(18-24)', '(25-34)', '(35-44)', '(45-54)', '(55-64)', '(65-74)', '(75+)']
    age = ageList[agePred[0].argmax()]

    # Update age and gender values
    age_gender['Age'] = age
    age_gender['Gender'] = gender

    # Write gender and age data to CSV file
    if not age_gender['Written']:
        csv_writer.writerow([gender, age])
        age_gender['Written'] = True

    # Display results on the frame
    label = "{},{}".format(gender, age)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    # Load pre-trained models
    faceProtoPath = "opencv_face_detector.pbtxt"
    faceModelPath = "opencv_face_detector_uint8.pb"
    ageProtoPath = "age_deploy.prototxt"
    ageModelPath = "age_net.caffemodel"
    genderProtoPath = "gender_deploy.prototxt"
    genderModelPath = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModelPath, faceProtoPath)
    ageNet = cv2.dnn.readNet(ageModelPath, ageProtoPath)
    genderNet = cv2.dnn.readNet(genderModelPath, genderProtoPath)

    # Open video capture for default camera (webcam)
    video_capture = cv2.VideoCapture(0)

    # Check if the CSV file exists
    csv_filename = 'results.csv'
    file_exists = os.path.isfile(csv_filename)

    # Open the CSV file to store results
    with open(csv_filename, mode='a', newline='') as file:
        csv_writer = csv.writer(file)

        # Write header only if the file doesn't exist
        if not file_exists:
            csv_writer.writerow(['Gender', 'Age'])

        # Set the maximum duration for the camera to be on (in seconds)
        max_duration = 5

        # Record the start time
        start_time = time.time()

        # Initialize age and gender values
        age_gender = {'Age': None, 'Gender': None, 'Written': False}

        # Initialize Tkinter window
        root = tk.Tk()
        root.title("Real-Time Face-Gender-Age Detection")

        # Create a label to display the video feed
        label = tk.Label(root)
        label.pack()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect faces
            faces = detect_faces(frame, faceNet)

            # Use multithreading for gender and age prediction
            threads = []
            for face in faces:
                thread = threading.Thread(target=process_face, args=(genderNet, ageNet, frame, face, csv_writer, age_gender))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Convert the frame to RGB and then to ImageTk format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)

            # Update the label with the new frame
            label.configure(image=frame)
            label.image = frame

            # Check if the maximum duration has elapsed
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_duration:
                print("Maximum duration reached. Turning off the camera.")
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update the Tkinter window
            root.update()

    # Release the video capture object and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()