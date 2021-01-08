# 先进行颜色提取 再阈值分割
import cv2 as cv
import numpy as np

def blur(image):
    print(image.shape)
    blurred = cv.pyrMeanShiftFiltering(image, 20, 100) # 均值偏移边缘保留滤波 去噪不破坏边缘
    cv.imshow("blur_image",blurred)
    #cv.imwrite("C:/Users/Administrator/Desktop/11.jpg", blurred)
    return blurred

def get_image_info(image):
    print("-----info-----")
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)

def color_detect(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 43, 46])  # HSV范围
    upper_hsv = np.array([34, 255, 255])
    mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))            #getStructuringElement()函数可用于构造一个特定大小和形状的结构元素，用于图像形态学处理
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=5)  # 连续 5 次进行闭操作
    colored = cv.bitwise_and(src1, src1, mask=mask)
    cv.imshow("mask", mask)
    #cv.imwrite("C:/Users/Administrator/Desktop/12.jpg", mask)
    cv.imshow("colored", colored)
    #cv.imwrite("C:/Users/Administrator/Desktop/13.jpg", colored)
    return colored

def ball_detect_demo(image,src):
    ball_detector = cv.CascadeClassifier("C:/Users/Administrator/Desktop/proj/xml/cascade.xml")
    balls = ball_detector.detectMultiScale(image, 1.02, 5)
    for x,y,w,h in balls:
        cv.rectangle(src, (x,y), (x+w, y+h),(0, 0 ,255), 2)
        print("x : %s,y : %s,w : %s,h : %s"%(x,y,w,h))
    cv.imshow("result", src)
    #cv.imwrite("C:/Users/Administrator/Desktop/15.jpg", src)

print("--------------hello python-------------")
src = cv.imread("D:/project/pic19.jpg")
src1 = src
#cv.namedWindow("input image",cv.WINDOW_AUTOSIZE) #这是因为添加了两个版本lib
cv.imshow("input image",src)
t1 = cv.getTickCount()
dst = blur(src)
t2 = cv.getTickCount()
#get_image_info(dst)
#get_image_info(src1)
dst2 = color_detect(dst)
t3 = cv.getTickCount()
dst_not = cv.bitwise_not(dst2)
dst_gray = cv.cvtColor(dst_not, cv.COLOR_BGR2GRAY)
cv.imshow("dst_gray",dst_gray)
#cv.imwrite("C:/Users/Administrator/Desktop/14.jpg", dst_gray)
t4 = cv.getTickCount()
ball_detect_demo(dst_gray,src1)
t5 = cv.getTickCount()
time1 = (t2-t1)/cv.getTickFrequency()
print("time1 : %s ms"%(time1*1000))  #计算时间
time2 = (t3-t2)/cv.getTickFrequency()
print("time2 : %s ms"%(time2*1000))  #计算时间
time3 = (t4-t3)/cv.getTickFrequency()
print("time3 : %s ms"%(time3*1000))  #计算时间
time4 = (t5-t4)/cv.getTickFrequency()
print("time4 : %s ms"%(time4*1000))  #计算时间

cv.waitKey(0)
cv.destroyAllWindows()