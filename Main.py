import cv2
import numpy as np
import threading
import time
from PIL import Image
from skimage.restoration import estimate_sigma
import os
import multiprocessing


def est_noise(image):
    return estimate_sigma(image, average_sigmas=True, multichannel=True)



def median():
    img = cv2.imread('Sample_BinaryImage_2.png', 1)
    median_blur = cv2.medianBlur(img, 3)
    cv2.imwrite('output1.jpg', median_blur)
    p=str(est_noise(median_blur))
    file=open("myfile.txt","a")
    file.write(p)
    file.write("\n")
    file.close()

def connected():
    src = cv2.imread('Sample_BinaryImage_2.png', cv2.IMREAD_GRAYSCALE)
    ret, binary_map = cv2.threshold(src, 127, 255, 0)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 10:
            result[labels == i + 1] = 255
    median_blur = cv2.medianBlur(result, 3)
    cv2.imwrite('output2.jpg', median_blur)
    p= str(est_noise(median_blur))
    file = open("myfile.txt", "a")
    file.write(p)
    file.write("\n")
    file.close()


def Gaussian():
    img = cv2.imread('Sample_BinaryImage_2.png', 0)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.threshold(blur, 90, 300, cv2.THRESH_BINARY)[1]
    cv2.imwrite('output3.jpg', thresh)
    p = str(est_noise(thresh))
    file = open("myfile.txt", "a")
    file.write(p)
    file.write("\n")
    file.close()


if __name__ == '__main__':
    total_time = 0.0
    start_timer = 0.0
    end_timer = 0.0

    # START TIMER
    start_timer = cv2.getTickCount()

    # ------------------------------------------------------------------------------------------------ #
    #                                          START CODE HERE                                         #
    # ------------------------------------------------------------------------------------------------ #
    threading.Thread(target=median).start()
    threading.Thread(target=connected).start()
    threading.Thread(target=Gaussian).start()
    #p1 = multiprocessing.Process(target=median)
    #p1.start()
    #p1.join()
    #p2 = multiprocessing.Process(target=connected)
    #p2.start()
    #p2.join()
    #p3 = multiprocessing.Process(target=Gaussian)
    #p3.start()
    #p3.join()

    A = []
    time.sleep(2)
    file = open("myfile.txt" , "r")
    for num in file:
        A.append(num)
        print(num)
    l=max(A[0],A[1],A[2])
    final=A.index(l)
    print(final)
    file.close()
    if os.path.exists("myfile.txt"):
        os.remove("myfile.txt")



    # Your code here

    # ------------------------------------------------------------------------------------------------ #
    #                                           END CODE HERE                                          #
    # ------------------------------------------------------------------------------------------------ #

    # END TIMER
    end_timer = cv2.getTickCount()

    if final==0:
        img1 = cv2.imread("output1.jpg")
        cv2.imshow('final', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if final==1:
        img2 = cv2.imread("output2.jpg")
        cv2.imshow('final', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if final==3:
        img3 = cv2.imread("output3.jpg")
        cv2.imshow('final', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    message = 'Time taken: {:0.4f} ms'
    total_time = (end_timer - start_timer) * 1000 / cv2.getTickFrequency()
    print(message.format(total_time))

    pass
