import cv2

# Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Input webcam
# 0 = webcam, type can be replaced any types (img, videos)
webcam = cv2.VideoCapture(0) 

# Show the current fame
while True:
    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # if there's an error, abort
    if not successful_frame_read:
        break
    
    # covert to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dectect faces + smiles
    # detecMultiScale = detect multi different face scales (small, big, etc)
    # Change to grayscale for speeding up dectection
    faces = face_detector.detectMultiScale(frame_grayscale)
    
    # Run faces detection within each of those faces
    for (x, y ,w, h) in faces:

        # Draw a rectangle around the face
        # rectangle (img, pt1, pt2, color, thickness)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the sub frame (using nympy N-dimensional array slicing)
        # frame[ y-axis, x-xis]
        the_face = frame[y:y+h, x:x+w]

        # covert face to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Run smiles detection within each of those faces
        # scaleFactor = how blur of image
        # minNeighbors = minimum combined rectangle boxes
        smiles = smile_detector.detectMultiScale(
        face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # for (x_, y_, w_, h_) in smiles:

        #     # Draw a rectangle around the face
        #     # rectangle (img, pt1, pt2, color, thickness)
        #     cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50 , 200), 4)

            # Label the smile
        if len(smiles) > 0:
            cv2.putText(frame, 'Dont smile', (x, y+h+40), fontScale=2,
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255 ,255))
    
    # Show the frame from webcam
    cv2.imshow('Hold your SMILE', frame)

    cv2.waitKey(1)

# cleanup webcam memory and close window    
webcam.release()
cv2.destroyAllWindows()

print('Detector running')