import numpy as np
from tqdm import tqdm
from itertools import chain as iterchain
from ..raw import CK
from ..raw import JAFFE
from ...paths import PATH_DATA_PROCESSED
from ...utils import *

def get_biometry(image, detector, predictor2d, predictor3d, transform):
    ok, rectangles, landmarks, facemarks = get_face_features(image, detector, predictor2d, predictor3d, transform)
    biometrics = get_face_biometrics(facemarks)
    biometry = biometrics[0].numpy()
    return biometry

def iter_preproc_datasets(**kwargs):
    dataset1 = CK(**kwargs)
    dataset2 = JAFFE(**kwargs)
    dataset = iterchain(dataset1, dataset2)
    detector = dlib_face_frontal_detector()
    predictor2d = dlib_face_landmarks_predictor()
    predictor3d = get_fm_depth_predictor()
    transform = get_landmarks_transform()
    for image, class_name in dataset:
        biometry = get_biometry(image, detector, predictor2d, predictor3d, transform)
        yield biometry, class_name
        image = np.fliplr(image)
        biometry = get_biometry(image, detector, predictor2d, predictor3d, transform)
        yield biometry, class_name

def iter_samples(**kwargs):
    datapath = PATH_DATA_PROCESSED.joinpath(f"BIOMETRY{kwargs['num_classes']}")
    datapath.mkdir(parents=True, exist_ok=True)
    dataset = iter_preproc_datasets(**kwargs)
    for index, sample in enumerate(tqdm(dataset)):
        yield (datapath, index) + sample

def save_sample(args):
    datapath, index, biometry, class_name = args
    savepath = datapath / class_name
    if not savepath.exists():
        savepath.mkdir(parents=True, exist_ok=True)
    filename = savepath.joinpath(f'{index}.npy').as_posix()
    np.save(filename, biometry)
    print('Saved:', filename, type(biometry))

def generate(**kwargs):
    return save_sample, iter_samples(**kwargs)
