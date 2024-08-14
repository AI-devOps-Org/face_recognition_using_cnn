import cv2
import face_recognition
import time

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("atharva2.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize variables
unlocked = False
webcam_open_time = 2  # Seconds
webcam_close_time = 2  # Seconds


def recognize_and_unlock():
    global unlocked
    # Start webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (OpenCV default) to RGB color
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

            if True in matches:
                unlocked = True
                break

        # If unlocked, show the video for 5 seconds and then break the loop
        if unlocked:
            print("Unlocked successfully!!")
            start_time = time.time()
            while time.time() - start_time < webcam_open_time:
                cv2.imshow('Video', frame)
                ret, frame = video_capture.read()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break
        else:
            print("Unknown face!!")
            start_time = time.time()
            while time.time() - start_time < webcam_close_time:
                cv2.imshow('Video', frame)
                ret, frame = video_capture.read()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            break
        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_and_unlock()
