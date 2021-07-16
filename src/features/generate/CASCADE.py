import cv2
import numpy as np
from tqdm import tqdm
from itertools import chain as iterchain
from ..raw import CK
from ..raw import JAFFE
from ...paths import PATH_DATA_PROCESSED
from ...utils import *

def get_features(image, detector, predictor2d, predictor3d, transform):
    ok, rectangles, landmarks, facemarks = get_face_features(image, detector, predictor2d, predictor3d, transform)
    biometrics = get_face_biometrics(facemarks)
    x, y, w, h = dlib_rectangle_to_numpy_box(image.shape, rectangles[0])
    size = (48, 48)
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    biometry = biometrics[0].numpy()
    return face, biometry

def iter_preproc_datasets(**kwargs):
    dataset1 = CK(**kwargs)
    dataset2 = JAFFE(**kwargs)
    dataset = iterchain(dataset1, dataset2)
    detector = dlib_face_frontal_detector()
    predictor2d = dlib_face_landmarks_predictor()
    predictor3d = get_fm_depth_predictor()
    transform = get_landmarks_transform()
    get_feature = lambda image: get_features(image, detector, predictor2d, predictor3d, transform)
    for image, class_name in dataset:
        face, biometry = get_feature(image)
        yield face, biometry, class_name
        image = np.fliplr(image)
        face, biometry = get_feature(image)
        yield face, biometry, class_name

def iter_samples(**kwargs):
    datapath = PATH_DATA_PROCESSED.joinpath(f"CASCADE{kwargs['num_classes']}")
    datapath.mkdir(parents=True, exist_ok=True)
    dataset = iter_preproc_datasets(**kwargs)
    for index, sample in enumerate(tqdm(dataset)):
        yield (datapath, index) + sample

def save_sample(args):
    datapath, index, image, biometry, class_name = args
    savepath = datapath / class_name
    if not savepath.exists():
        savepath.mkdir(parents=True, exist_ok=True)
    filename = savepath.joinpath(f'{index}.png').as_posix()
    cv2.imwrite(filename, image)
    print('Saved:', filename)
    filename = savepath.joinpath(f'{index}.npy').as_posix()
    np.save(filename, biometry)
    print('Saved:', filename, type(biometry))

def generate(**kwargs):
    return save_sample, iter_samples(**kwargs)
