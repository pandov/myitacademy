import dlib
dlib.DLIB_USE_CUDA = True
import numpy as np
from src.utils import DIR

def dlib_rectangle(rectangle):
    if type(rectangle) is dlib.mmod_rectangle:
        rectangle = rectangle.rect
    return rectangle

def dlib_rectangle_sorter(rectangle):
    rectangle = dlib_rectangle(rectangle)
    return (rectangle.right() - rectangle.left()) * (rectangle.bottom() - rectangle.top())

def dlib_face_frontal_detector():
    return dlib.get_frontal_face_detector()

def dlib_cnn_face_detector():
    return dlib.cnn_face_detection_model_v1(DIR.MODELS.joinpath('mmod_human_face_detector.dat').as_posix())

def dlib_face_landmarks_predictor():
    return dlib.shape_predictor(DIR.MODELS.joinpath('shape_predictor_68_face_landmarks.dat').as_posix())

def dlib_rectangle_to_numpy_box(shape, rectangle):
    h, w = shape[:2]
    rectangle = dlib_rectangle(rectangle)
    x, y = rectangle.left(), rectangle.top()
    return np.array([max(x, 0), max(y, 0), min(rectangle.right(), w), min(rectangle.bottom(), h)])

def dlib_crop_rectangle_from_image(image, rectangle):
    x1, y1, x2, y2 = dlib_rectangle_to_numpy_box(image.shape, rectangle)
    crop = image[y1:y2, x1:x2]
    return crop

def dlib_shape_to_numpy_array(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(len(shape.parts()))])
