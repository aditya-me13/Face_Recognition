import os
import sys
import face_recognition
import cv2
import numpy as np
import math
import logging

# Clear the logging file before execution
with open('logs.log', 'w'):
    pass

logging.basicConfig(
    filename='logs.log',  # Log file name
    level=logging.INFO,      # Log level
    format='%(asctime)s %(name)s %(levelname)s %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)
logger = logging.getLogger('main.py')


def face_conidence(face_distance, face_match_threshold=0.8):
    _range = (1.0 - face_match_threshold)
    linear_value = (1.0 - face_distance) / (_range * 2.0)

    if(face_distance > face_match_threshold):
        return str(round(linear_value*100, 2)) + '%'
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value-0.5)*2,0.2))) * 100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        logger.info('FaceRecognition class initialized')
        self.encode_faces()
    
    def encode_faces(self):
        logger.info('Encoding faces')
        for image in os.listdir('faces'): # list of items in faces folder (with extension)
            face_image = face_recognition.load_image_file(f'faces/{image}') # read in RBG or other numerical method
            face_encoding = face_recognition.face_encodings(face_image)[0] # encode it into numbers starting from top left

            self.known_face_encodings.append(face_encoding) # add encoding
            self.known_face_names.append(image.split('.')[0]) # add corresponding Name

        logger.info('Faces encoded')

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print('Unable to load camera.')
            sys.exit()
        
        logger.info('Camera loaded')
        while True:
            # ret returns if theres is any frame to process (Bool value)
            ret, frame = video_capture.read() 
            
            # Process every alternate frame
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # compress the frame to 0.25 to speed up processing
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) # think of a blackbox that converts to RGB format

                # find all faces in the current frame
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations) 

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # see if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.8)
                    name = "Unknown"
                    confidence = 'Unknown'

                    face_distance = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distance)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_conidence(face_distance[best_match_index])
                        logger.info(f'Face {name} is {confidence}')

                    self.face_names.append(f'{name} ({confidence})')
                    
            self.process_current_frame = not self.process_current_frame

            # display annotations
            for (top, right, bottom, left), name in zip(reversed(self.face_locations), reversed(self.face_names)):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # create a rectange with the boundaries and color = red (0, 0, 255) and thickness 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # create a text with the name and color = white (255, 255, 255)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        logger.info('Program finished')


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
