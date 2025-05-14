import cv2
import numpy as np
import os
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow own logs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TF v1 deprecation warnings


# Function to get the absolute path of the model
def get_absolute_path(path):
    # Remove leading and trailing quotes if they exist
    path = path.strip('"')
    abs_path = os.path.abspath(path)  # Convert the input path to an absolute path
    print(f"Model path: {abs_path}")  # print the absolute path
    return abs_path

# Input paths
model = get_absolute_path(r"C:\Users\Naveen\Documents\Information Technology\Handwritten Number Recognition\Code\model.h5")  # model path
print("Model Selected Successfully")

refimg = get_absolute_path(r"C:\Users\Naveen\Documents\Information Technology\Handwritten Number Recognition\Code\reference_image.png")  # reference image path
print("Reference Image Selected Successfully")

inpimg = get_absolute_path(r"C:\Users\Naveen\Documents\Information Technology\Handwritten Number Recognition\Code\input_images")  # input images folder path
print("Input Images Folder Selected Successfully")



roi_values = [[(466, 1220), (764, 1302), 'text', 'a1'],
            [(776, 1226), (1088, 1306), 'text', 'a2'],
            [(1100, 1230), (1424, 1310), 'text', 'a3'],
            [(470, 1310), (764, 1392), 'text', 'a4'],
            [(778, 1318), (1084, 1396), 'text', 'a5'],
        [(1100, 1322), (1422, 1400), 'text', 'a6'],
        [(472, 1396), (766, 1476), 'text', 'b6a1'],
        [(776, 1404), (1086, 1482), 'text', 'b8a2'],
        [(1100, 1408), (1424, 1488), 'text', 'b6a3'],
        [(472, 1486), (764, 1564), 'text', 'b6b1'],
        [(776, 1494), (1084, 1570), 'text', 'b6b2'],
        [(1098, 1498), (1424, 1576), 'text', 'b6b3'],
        [(472, 1570), (766, 1650), 'text', 'b7a1'],
        [(776, 1576), (1088, 1656), 'text', 'b7a2'],
        [(1098, 1582), (1424, 1668), 'text', 'b7a3'],
        [(474, 1660), (766, 1738), 'text', 'b7b1'],
        [(776, 1666), (1088, 1746), 'text', 'b7b2'],
        [(1100, 1674), (1426, 1756), 'text', 'b7b3'],
        [(474, 1746), (766, 1828), 'text', 'c8a1'],
        [(778, 1752), (1086, 1836), 'text', 'c8a2'],
        [(1100, 1758), (1426, 1846), 'text', 'c8a3'],
        [(474, 1836), (764, 1918), 'text', 'c8b1'],
        [(778, 1842), (1088, 1928), 'text', 'c8b2'],
        [(1100, 1850), (1428, 1944), 'text', 'c8b3']]


"""**********************************************************************************"""
def start_program():

    new_model = load_model(model)
    imgq = cv2.imread(refimg)
    path = inpimg
    roi = roi_values

    #new_model.summary()
    def test_pipeline_equation_offline(image_path):
        chars = []
        img = image_path
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edged = cv2.Canny(img_gray, 30, 150)
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sort_contours(contours, method="left-to-right")[0]
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if 20<=w and 30<=h:
                roi = img_gray[y:y+h, x:x+w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (th, tw) = thresh.shape
                if tw > th:
                    thresh = imutils.resize(thresh, width=32)
                if th > tw:
                    thresh = imutils.resize(thresh, height=32)
                (th, tw) = thresh.shape
                dx = int(max(0, 32 - tw)/2.0)
                dy = int(max(0, 32 - th) / 2.0)
                padded = cv2.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))
                padded = cv2.resize(padded, (32, 32))
                padded = np.array(padded)
                padded = padded/255.
                padded = np.expand_dims(padded, axis=0)
                padded = np.expand_dims(padded, axis=-1)
                pred = new_model.predict(padded)
                pred = np.argmax(pred, axis=1)
                print(pred)
                label = labels[pred[0]]
                chars.append(label)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, label, (x-5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        figure = plt.figure(figsize=(10, 10))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()
    
        #print(chars)

        my_list = chars
        result = str(''.join(map(str, my_list)))
    
        return(result)

        #print(result)
        #print(type(result))

    per = 50;
    h,w,c = imgq.shape
    orb = cv2.ORB_create(10000)                                                        #it defines features defaault is 500 we put 1000 we can change based on our needs
    kp1, des1 = orb.detectAndCompute(imgq,None)
    impkp1 = cv2.drawKeypoints(imgq,kp1,None)
    mypiclist = os.listdir(path)
    print(mypiclist)
    for j,y in enumerate(mypiclist):
        img = cv2.imread(path +"/"+y)
        #img = cv2.resize(imgq,(w//3,h//3))
        #cv2.imshow(y, img)                                                         #individual file printing

        #finding descriptors and finding the matches
        kp2, des2 = orb.detectAndCompute(img,None)                                  #here the "img" variable contains all the user input images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)                                        # we are using brute force matcher details see in documantion
        matches = bf.match(des2,des1)
        matches = sorted(matches, key=lambda x: x.distance)                         #lambda is a single line function that sorts based on the distance
        good = matches[:int(len(matches)*(per/100))]                                #gives the best matches with value from the per variable
        imgmatch = cv2.drawMatches(img,kp2,imgq,kp1,good[:100],None,flags=2)        #it gives the best 20 matches
        #imgmatch = cv2.resize(imgmatch,(w//3,h//3))
        #cv2.imshow(y, imgmatch)

        #align the images based on the source image

        srcpoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)       #source points
        dstpoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1 ,1, 2)     #dest points
    
        M, _ =cv2.findHomography(srcpoints,dstpoints,cv2.RANSAC,5.0)                     #finding relationship is called homography
        imgscan = cv2.warpPerspective(img,M,(w,h))
    
        #cv2.imshow(y, imgscan)

        imgshow = imgscan.copy()
        imgmask = np.zeros_like(imgshow)

        mydata = []

        for x,r in enumerate(roi):
        
            cv2.rectangle(imgmask, ((r[0][0]),r[0][1]),((r[1][0]),r[1][1]),(0,255,0),cv2.FILLED)
            #imgshow = cv2.addWeighted(imgshow,0.99,imgmask,0.1,0)                                   #the area to be cropped that is sent to pytesseract is done here
            
            imgcrop = imgscan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
            #cv2.imshow(str(x), imgcrop)
           
            #print(f'{r[3]} :{test_pipeline_equation_offline(imgcrop)}')
            #mydata.append(test_pipeline_equation_offline(imgcrop))
        
            #test_pipeline_equation_offline(imgcrop)

            if r[2] == 'text':
                #cv2.imshow(str(x), imgcrop)
                #print(f'{r[3]} :{test_pipeline_equation_offline(imgcrop)}')
                mydata.append(test_pipeline_equation_offline(imgcrop))
        print(mydata)

        with open('output.csv','a+') as f:
            for data in mydata:
                f.write((str(data)+','))
            f.write('\n')

start_program()




