import numpy as np
import cv2 as cv

def load_yolo_classes():
    classes = []
    with open('coco.names','r') as f:
        classes = f.read().splitlines()
    return classes

def yolo_result(blob,height,width):   
    net = cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.75:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width) 
                h = int(detection[3]*height)

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes,confidences,class_ids

            


def pre_process(img):
    blob = cv.dnn.blobFromImage(img, 1/255, (416,416), swapRB =True)
    height, width, _ = img.shape
    boxes,confidences,class_ids = yolo_result(blob,height, width)
    return boxes,confidences,class_ids



cap = cv.VideoCapture('xyz.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    height, width, _ = frame.shape

    boxes,confidences,class_ids =pre_process(frame)
    classes =load_yolo_classes()

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

    font = cv.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,225,size=(len(boxes),3))

    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = "{}: {:.4f}".format(classes[class_ids[i]],
            confidences[i])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv.rectangle(frame, (x,y), (x+w, y+h), color,2)
            cv.putText(frame, label +" "+confidence, (x, y+20), font, 2,(0,255,0),2)
    
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


