import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet("yolov3_training.cfg",r"yolov3_training_last.weights")
classes = ['pass','er01','er02']
def show_results(image):

    while True:

        hight,width,_ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

        net.setInput(blob)

        output_layers_name = net.getUnconnectedOutLayersNames()

        layerOutputs = net.forward(output_layers_name)

        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.6:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3]* hight)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)
        font = cv2.FONT_HERSHEY_PLAIN
        if  len(indexes)>0:
            for i in indexes.flatten(): # bổ sung

                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                if label == 'pass':
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0, 0, 255),1)
                    cv2.rectangle(image, (x, y), (x + w, y - 15),(0, 0, 255), -1)
                    cv2.putText(image,label + " " + confidence, (x,y-5),font,1,(255, 255, 255),1)
                elif label == 'er01':
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 255),2)
                    cv2.putText(image,label + " " + confidence, (x,y-5),font,1,(0, 255, 255),2)                
                elif label == 'er02':
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255, 255),2)
                    cv2.putText(image,label + " " + confidence, (x,y-5),font,1,(255, 255, 255),2)  
            return image    
        return image

def identify(frame): 
	return show_results(image=frame)

def display_frames_per_second(frame, start_time):
	END_TIME = abs(start_time-time.time())
	TOP_LEFT = (0,0)
	BOTTOM_RIGHT = (116,26)
	TEXT_POSITION = (8,20)
	TEXT_SIZE = 0.6
	FONT = cv2.FONT_HERSHEY_SIMPLEX
	COLOR = (255,255,0) #BGR
	# cv2.rectangle(frame, TOP_LEFT, BOTTOM_RIGHT, (0,0,0), cv2.FILLED)
	cv2.putText(frame, "FPS: {}".format(round(1/max(0.0333,END_TIME),1)), TEXT_POSITION, FONT, TEXT_SIZE,COLOR)
	return frame