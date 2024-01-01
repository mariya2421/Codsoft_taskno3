import cv2

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haarcascade classifier for face detection
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the LBPH face recognizer and read the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# Define a list of names corresponding to the recognized face IDs
# Assuming you have labeled your training data starting from 0
name_list = ["", "","malu","Mariya"]

while True:
    # Read a frame from the video capture
    ret, frame = video.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Recognize the face using the trained model
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if the confidence is below a certain threshold
        if conf < 50 and 0 <= serial < len(name_list):
            # Display the recognized name and draw a rectangle around the face
            cv2.putText(frame, name_list[serial], (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        else:
            # If confidence is high or serial is out of range, label the face as "Unknown"
            cv2.putText(frame, "Unknown", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with face recognition results
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to exit the loop
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

print("face identified")
