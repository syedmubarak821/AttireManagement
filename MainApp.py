from unittest import result
import numpy as np
import cv2
from keras.models import load_model

labels_dict={0:'Not Allowed',1:'Allowed'}
color_dict={0:(0,0,255),1:(0,255,0)}
size1 = 2
j=0
Body_cascade=""
def Predictor(img,Body_cascade,model):
    mini = cv2.resize(img, (img.shape[1] // size1, img.shape[0] // size1))
    Detection = Body_cascade.detectMultiScale(mini)
    label=''
    for f in Detection:
        (x, y, w, h) = [v * size1 for v in f]
        Detection_img = img[y:y+h, x:x+w]
        resized=cv2.resize(Detection_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        #print(result)  
        #print(labels_dict[label]);
    
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        return label

def open_webcam():
    webcam = cv2.VideoCapture(0)
    allowed=0
    notallowed=0
    Body_cascade1 = cv2.CascadeClassifier('C:/Users/Syed Mubarak/AppData/Local/Programs/Python/Python310/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')    
    model1=load_model("TrainedModel/model2-003.model")
    Body_cascade2 = cv2.CascadeClassifier('C:/Users/Syed Mubarak/AppData/Local/Programs/Python/Python310/lib/site-packages/cv2/data/haarcascade_upperbody.xml')    
    model2=load_model("TrainedModel/upperbody-003.model")
    Body_cascade3=cv2.CascadeClassifier('C:/Users/Syed Mubarak/AppData/Local/Programs/Python/Python310/lib/site-packages/cv2/data/haarcascade_lowerbody.xml')    
    model3=load_model("TrainedModel/LowerBody-001.model")
    while True:
        (rval, img1) = webcam.read()
        img=cv2.flip(img1,1,1)
        outputval=(Predictor(img,Body_cascade1,model1))
        if(outputval == 1):
            allowed+=1;
        elif(outputval == 0):
            notallowed+=1
        outputval=(Predictor(img,Body_cascade2,model2))
        if(outputval == 1):
            allowed+=1;
        elif(outputval == 0):
            notallowed+=1
        outputval=(Predictor(img,Body_cascade3,model3))
        if(outputval == 1):
            allowed+=1;
        elif(outputval == 0):
            notallowed+=1
        cv2.imshow('LIVE',   img)
        key = cv2.waitKey(10)
        if key == 27: #The Esc key
            break
    # Stop video
    webcam.release()
    cv2.imshow('image',img)
    cv2.destroyAllWindows()
    if(allowed >= notallowed):
        return 1
    else:
        return 0

if (open_webcam()):
    print("Welcome!\nYou Are Allowed")
else:
    print("Sorry!\nDress Code Doesnot Match The Office Dressing Rule\n");
