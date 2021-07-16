import torch
from src.utils import DIR
from src.nn import *
from .connections import connections
from .dlib import *

def get_fm_depth_predictor(device='cpu'):
    model = FMDepthRNN().eval().to(device)
    state_dict = torch.load(DIR.MODELS.joinpath('fmdepthrnn.pth'))['model_state_dict']
    model.load_state_dict(state_dict)
    return model

def get_normalized_landmarks(batch):
    batch = torch.tensor(batch, dtype=torch.float32)
    tmin = torch.min(batch, dim=1).values
    tmax = torch.max(batch, dim=1).values
    length = batch.shape[0]
    center = torch.div(tmin + tmax, 2).reshape(length, 1, -1)
    shape = torch.div(tmax - tmin, 2).reshape(length, 1, -1)
    batch = torch.sub(batch, center)
    batch = torch.div(batch, shape)
    return batch

def get_facemarks_3d(landmarks, predictor, device='cpu'):
    '''
    predictor = get_fm_depth_predictor()
    '''
    landmarks = get_normalized_landmarks(landmarks).to(device)
    depth = predictor(landmarks).detach().unsqueeze(2)
    facemarks = torch.cat((landmarks, depth), dim=2)
    return facemarks.cpu()

def get_facial_biometrics(facemarks):
    '''
    facemarks = get_facemarks_3d(...)
    '''
    idx1 = connections[:, 0].tolist()
    idx2 = connections[:, 1].tolist()
    pts1 = facemarks[:, idx1, :]
    pts2 = facemarks[:, idx2, :]
    dist = torch.sqrt(((pts1 - pts2) ** 2).sum(axis=2))
    return dist

def get_facial_features(image, detector, predictor2d, predictor3d, upsample_num_times):
    '''
    detector = dlib_face_frontal_detector()
    detector = dlib_cnn_face_detector()
    predictor2d = dlib_face_landmarks_predictor()
    predictor3d = get_fm_depth_predictor()
    '''
    rectangles = dlib_facial_rectangles(image, detector, upsample_num_times)
    if len(rectangles) == 0:
        return False, [], [], []
    landmarks = dlib_facial_landmarks(image, rectangles, predictor2d)
    facemarks = get_facemarks_3d(landmarks, predictor3d)
    return True, rectangles, landmarks, facemarks
