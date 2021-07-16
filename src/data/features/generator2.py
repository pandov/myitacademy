import cv2
import numpy as np
import os
from itertools import chain as iterchain
from multiprocessing import Pool
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

class Generator(object):
    def __init__(self):
        self.FER = DIR.DATA.PROCESSED.joinpath('FER')
        self.BIOMETR = DIR.DATA.PROCESSED.joinpath('BIOMETR')

    def __iter__(self):
        detector = dlib_face_frontal_detector()
        predictor2d = dlib_face_landmarks_predictor()
        predictor3d = get_fm_depth_predictor()
        dataset = iterchain(CK(), JAFFE(), RAFDB())
        for index, (image, class_name) in enumerate(tqdm(dataset)):
            filename = f'{class_name}/{index}'
            ok, rectangles, _, facemarks = get_facial_features(image, detector, predictor2d, predictor3d, 1)
            if not ok: continue
            png = self.FER.joinpath(filename + '.png')
            if not png.exists():
                faces = dlib_crop_faces_from_image(image, rectangles[[0]])
                png.parent.mkdir(parents=True, exist_ok=True)
                yield png.as_posix(), faces[0]
            npy = self.BIOMETR.joinpath(filename + '.npy')
            if not npy.exists():
                biometrics = get_facial_biometrics(facemarks[[0]]).numpy()
                npy.parent.mkdir(parents=True, exist_ok=True)
                yield npy.as_posix(), biometrics[0]
            # if index >= 10: break

    def iter_spec(self):
        for index, (image, class_name) in enumerate(tqdm(iter(FER2013()))):
            filename = f'{class_name}/0{index}.png'
            filepath = self.FER.joinpath(filename)
            # if filepath.exists(): continue
            # filepath.parent.mkdir(parents=True, exist_ok=True)
            yield filepath.as_posix(), image
            # if index >= 10: break

    def generate(self, num_workers):
        # with Pool(num_workers) as pool:
        #     pool.map(save_sample, iter(self))
        with Pool(num_workers) as pool:
            return pool.map(save_sample, self.iter_spec())

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()
    Generator().generate(num_workers=args.num_workers)
