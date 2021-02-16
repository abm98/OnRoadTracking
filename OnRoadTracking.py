import cv2
from random import randrange

#Our Image
car_img = 'car2.jpg'
# video = cv2.VideoCapture('tesla accident.mp4')
video = cv2.VideoCapture('pedestrian.mp4')

#Our pre trained classifier data
car_classifier_file = 'cars.xml'
pedestrian_classifier_file = 'haarcascade_fullbody.xml'

#Create Car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
#Create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

#Run forever until car stops or something or crashes
while True:
    #Read car frame
    (read_successful, frame) = video.read()
    # safe coding
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #Detect car and Pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame) ##,scaleFactor = 1.1, minNeighbours=2)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame) #,scaleFactor = 1.1, minNeighbours=2)

    # Draw rectangle around cars
    for (x, y, w, h) in cars:
        # (0,0,0) change for color / ,4) change thickness
        cv2.rectangle(frame,(x+1,y+2),(x+w, y+h), (255,0,255),4)
        cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0),4)
    # Draw rectangle around pedestrians
    for (x, y, w, h) in pedestrians:
        # (0,0,0) change for color / ,4) change thickness
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,255),4)


    # Display the image spotted
    cv2.imshow('Pirated Self Driving Car', frame)
    # Don't Autoclose (wait for key to pressed to quit)
    key = cv2.waitKey(1)

        # STOP if Q(81/113 ASCII) key is pressed
    if key == 81 or key ==113:
        break
# Release the video object which is capture
video.release()

