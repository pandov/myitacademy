import torch
import torchvision.transforms.functional as TF
import cv2
from src.utils import IMG_SIZE
from .wrapper import *

def dlib_facial_rectangles(image, detector, upsample_num_times=0):
    '''
    detector = dlib_face_frontal_detector()
    detector = dlib_cnn_face_detector()
    '''
    rectangles = detector(image, upsample_num_times)
    rectangles = sorted(rectangles, key=dlib_rectangle_sorter, reverse=True)
    rectangles = np.array(rectangles)
    return rectangles

def dlib_facial_landmarks_lazy(image, rectangles, predictor):
    '''
    predictor = dlib_face_landmarks_predictor()
    '''
    for rectangle in rectangles:
        rectangle = dlib_rectangle(rectangle)
        landmarks = predictor(image, rectangle)
        landmarks = dlib_shape_to_numpy_array(landmarks)
        yield landmarks

def dlib_facial_landmarks(image, rectangles, predictor):
    '''
    predictor = dlib_face_landmarks_predictor()
    '''
    lazy = dlib_facial_landmarks_lazy(image, rectangles, predictor)
    landmarks = np.asarray(list(lazy))
    return landmarks

def dlib_crop_faces_from_image_lazy(image, rectangles):
    for rectangle in rectangles:
        crop = dlib_crop_rectangle_from_image(image, rectangle)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, IMG_SIZE)
        yield crop

def dlib_crop_faces_from_image(image, rectangles):
    lazy = dlib_crop_faces_from_image_lazy(image, rectangles)
    images = np.asarray(list(lazy))
    return images
