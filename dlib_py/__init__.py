# 모듈
import dlib
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
threshold = 0.5


def plotPairs(img1, img2):
    fig = plt.figure(figsize=(4,4))

    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")

    ax1 = fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")



def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def verify(img1_path, img2_path):
    img1 = dlib.load_rgb_image(img1_path)
    img2 = dlib.load_rgb_image(img2_path)
    img1_detection = detector(img1, 1)
    img2_detection = detector(img2, 1)

    if len(img1_detection) == 0:
        return False
        raise ValueError("no face detected in img1")

    if len(img2_detection) == 0:
        return False
        raise ValueError("no face detected in img2")

    img1_shape = sp(img1, img1_detection[0])
    img2_shape = sp(img2, img2_detection[0])

    img1_aligned = dlib.get_face_chip(img1, img1_shape)
    img2_aligned = dlib.get_face_chip(img2, img2_shape)

    img1_representation = facerec.compute_face_descriptor(img1_aligned)
    img2_representation = facerec.compute_face_descriptor(img2_aligned)

    img1_representation = np.array(img1_representation)
    img2_representation = np.array(img2_representation)

    distance = findEuclideanDistance(img1_representation, img2_representation)
    if distance < threshold:
        verified = True
        plotPairs(img1_aligned, img2_aligned)
    else:
        verified = False

    return verified


def determine(detect_img, detected_img):
    for i in range(len(detected_img)):
        for j in range(len(detect_img)):
            print(i, j)
            verified = verify(detect_img[j], detected_img[i])
            if verified:
                plt.savefig(f"static/dlib_result_image/detection{i, j}.jpg")
    
    # plt.close()
                


