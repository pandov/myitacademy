import cv2
from tqdm import tqdm
from itertools import chain as iterchain
from ..raw import CK
from ..raw import JAFFE
from ..raw import FER2013
from ...paths import PATH_DATA_PROCESSED
from ...utils import *

def iter_preproc_datasets(**kwargs):
    dataset1 = CK(**kwargs)
    dataset2 = JAFFE(**kwargs)
    dataset = iterchain(dataset1, dataset2)
    size = (48, 48)
    detector = dlib_face_frontal_detector()
    for image, class_name in dataset:
        rectangles = get_faces_rectangles(image, detector)
        rectangle = rectangles[0]
        x, y, w, h = dlib_rectangle_to_numpy_box(image.shape, rectangle)
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, size)
        yield image, class_name

def iter_fer_2013(**kwargs):
    return iter(FER2013(**kwargs))

def iter_samples(**kwargs):
    datapath = PATH_DATA_PROCESSED.joinpath(f"FER{kwargs['num_classes']}")
    datapath.mkdir(parents=True, exist_ok=True)
    dataset = iterchain(iter_preproc_datasets(**kwargs), iter_fer_2013(**kwargs))
    for index, sample in enumerate(tqdm(dataset)):
        yield (datapath, index) + sample

def save_sample(args):
    datapath, index, image, class_name = args
    savepath = datapath / class_name
    if not savepath.exists():
        savepath.mkdir(parents=True, exist_ok=True)
    filename = savepath.joinpath(f'{index}.png').as_posix()
    cv2.imwrite(filename, image)
    print('Saved:', filename)

def generate(**kwargs):
    return save_sample, iter_samples(**kwargs)
