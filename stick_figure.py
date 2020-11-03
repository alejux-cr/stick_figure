import numpy as np
import cv2

# GLOBAL VARIABLES 

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Methods
def run():
  cap = cv2.VideoCapture(0)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    for i in range(faces.shape[2]):
      confidence = faces[0, 0, i, 2]
      if confidence > 0.5:
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
  
    cv2.imshow('Stick Figure', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()


# Start program
def main():
  print("Initializing body tracking application...")
  print("Start moving when the video capture pops up :)")
  print("Press Q to quit the program")
  run()


if __name__ == '__main__':
  main()