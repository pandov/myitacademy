import cv2
import numpy as np
from itertools import chain as iterchain
from multiprocessing import Pool, Manager, Queue
from tqdm import tqdm
from src.data.external import *
from src.utils import DIR
from src.utils.contrib import *

def save_sample(args):
    filepath, data = args
    if '.png' in filepath:
        cv2.imwrite(filepath, data)
    elif '.npy' in filepath:
        np.save(filepath, data)
    print('Saved:', filepath)

def preproc_sample(args):
    filepath, imagepath = args
    queue = preproc_sample.queue
    detector, predictor2d, predictor3d = queue.get()
    queue.put((detector, predictor2d, predictor3d))
    image = cv2.imread(imagepath)
    rectangles = dlib_facial_rectangles(image, detector, 1)
    if len(rectangles) > 0:
        rectangles = rectangles[[0]]
        if '.png' in filepath:
            faces = dlib_crop_faces_from_image(image, rectangles)
            save_sample((filepath, faces[0]))
            del faces
        elif '.npy' in filepath:
            landmarks = dlib_facial_landmarks(image, rectangles, predictor2d)
            facemarks = get_facemarks_3d(landmarks, predictor3d)
            biometrics = get_facial_biometrics(facemarks).numpy()
            save_sample((filepath, biometrics[0]))
            del landmarks, facemarks, biometrics
    del image, rectangles

def init_queue(queue):
    preproc_sample.queue = queue

class Generator(object):
    def __init__(self, path, ext):
        self.datapath = DIR.DATA.PROCESSED.joinpath(path)
        self.ext = ext

    def __iter__(self):
        dataset = iterchain(CK(), JAFFE(), RAFDB())
        for index, (imagepath, class_name) in enumerate(tqdm(dataset)):
            filename = f'{class_name}/{index}.{self.ext}'
            filepath = self.datapath.joinpath(filename)
            if filepath.exists(): continue
            parent = filepath.parent
            if not parent.exists(): parent.mkdir(parents=True, exist_ok=True)
            yield filepath.as_posix(), imagepath

    def generate(self, num_workers):
        queue = Queue()
        for i in range(num_workers):
            detector = dlib_face_frontal_detector()
            predictor2d = dlib_face_landmarks_predictor()
            predictor3d = get_fm_depth_predictor()
            queue.put((detector, predictor2d, predictor3d))
        with Pool(num_workers, init_queue, [queue]) as pool:
            return pool.map(preproc_sample, iter(self))


class GeneratorFER(Generator):
    def __init__(self, **kwargs):
        super().__init__('FER', 'png', **kwargs)

    def iter_spec(self):
        for index, (image, class_name) in enumerate(iter(FER2013())):
            filename = f'{class_name}/0{index}.{self.ext}'
            filepath = self.datapath.joinpath(filename)
            if filepath.exists(): continue
            yield filepath.as_posix(), image

    def generate(self, num_workers):
        super().generate(num_workers)
        with Pool(num_workers) as pool:
            return pool.map(save_sample, self.iter_spec())


class GeneratorBIOMETR(Generator):
    def __init__(self, **kwargs):
        super().__init__('BIOMETR', 'npy', **kwargs)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--FER', action='store_true')
    parser.add_argument('--BIOMETR', action='store_true')
    parser.add_argument('--CASCADE', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    kwargs = dict(num_workers=args.num_workers)
    if args.FER:
        GeneratorFER().generate(**kwargs)
    elif args.BIOMETR:
        GeneratorBIOMETR().generate(**kwargs)
