import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
model = tf.keras.models.load_model('Digit_recong.h5')

a = np.ones([300,300],dtype='uint8')*255
cv2.rectangle(a,(50,50),(250,250),(0,0,0),-5)
print("Press p for Predictation")
print("Press c for Clear")
print("Press Esc for Quit")
wname = 'Digits'
stat = False
cv2.namedWindow(wname)

def digits(event,x,y,flags,param):
    global stat
    if event == cv2.EVENT_LBUTTONDOWN:
        stat = True
    elif (event == cv2.EVENT_MOUSEMOVE):
        if (stat==True):
            cv2.circle(a,(x,y),5,(255,255,255),-3)
    elif event== cv2.EVENT_LBUTTONUP:
        stat = False

cv2.setMouseCallback(wname,digits)

while True:
    cv2.imshow(wname,a)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('p'):
        digit = a[50:250, 50:250]
        #cv2.imshow("Cropped",digit)
        digit = digit/255
        digit = cv2.resize(digit,(28,28)).reshape(1,28,28)
        print("The digit recognized is:",np.argmax(model.predict(digit)))
    elif key == ord('c'):
        a[:,:] = 255
        cv2.rectangle(a,(50,50),(250,250),(0,0,0),-5)
cv2.destroyAllWindows()