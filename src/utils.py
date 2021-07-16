import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms
from .connections import connections
from .dlib_wrapper import *
from .models import *
from .transforms import *

def get_fm_depth_predictor(**kwargs):
    model = FMDepthRNN(**kwargs)
    model.init_weights()
    model.eval()
    return model

def get_fc_fer_predictor(**kwargs):
    model = FERFC(**kwargs)
    model.init_weights()
    model.eval()
    return model

def get_cnn_fer_predictor(**kwargs):
    model = FERCNN(**kwargs)
    model.init_weights()
    model.eval()
    return model

def get_cascade_fer_predictor(**kwargs):
    model = FERNet(**kwargs)
    model.init_weights()
    model.eval()
    return model

def get_faces_rectangles(image, detector, upsample_num_times=0):
    # detector = dlib_face_frontal_detector()
    rectangles = detector(image, upsample_num_times)
    rectangles = sorted(rectangles, key=dlib_rectangle_sorter, reverse=True)
    rectangles = np.asarray(rectangles)
    return rectangles

def get_lazy_face_landmarks_2d(image, rectangles, predictor):
    # predictor = dlib_face_landmarks_predictor()
    for rectangle in rectangles:
        rectangle = dlib_rectangle(rectangle)
        landmarks = predictor(image, rectangle)
        landmarks = dlib_shape_to_numpy_array(landmarks)
        yield landmarks

def get_face_landmarks_2d(image, rectangles, predictor):
    lazy = get_lazy_face_landmarks_2d(image, rectangles, predictor)
    landmarks = np.asarray(list(lazy))
    return landmarks

def get_landmarks_transform():
    return transforms.Compose([
        FaceLandmarksNormalize(),
        torch.FloatTensor,
    ])

def get_facemarks_3d(landmarks, transform, predictor):
    # transform = get_landmarks_transform()
    landmarks = transform(landmarks)
    # predictor = get_fm_depth_predictor()
    depth = predictor(landmarks).detach().unsqueeze(2)
    facemarks = torch.cat((landmarks, depth), dim=2)
    return facemarks

def get_face_biometrics(facemarks):
    idx1 = connections[:, 0].tolist()
    idx2 = connections[:, 1].tolist()
    pts1 = facemarks[:, idx1, :]
    pts2 = facemarks[:, idx2, :]
    dist = torch.sqrt(((pts1 - pts2) ** 2).sum(axis=2)) # distance
    return dist

def get_lazy_face_images(image, rectangles, transform):
    for rectangle in rectangles:
        x, y, w, h = dlib_rectangle_to_numpy_box(image.shape, rectangle)
        part = image[y:y+h, x:x+w]
        if transform:
            part = transform(Image.fromarray(part))
        yield part

def get_face_images(image, numpy_rectangles, transform=None):
    lazy = get_lazy_face_images(image, numpy_rectangles, transform)
    images = list(lazy)
    if transform:
        images = torch.stack(images)
    else:
        images = np.asarray(images)
    return images

def get_face_images_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

def get_fer_names(num_classes):
    names = {
        5: ['anger', 'fear', 'happy', 'sadness', 'surprise'],
            6: ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
            7: ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
    }
    return names.get(num_classes)

def get_fc_weights(num_classes, weights):
    weights = torch.FloatTensor(weights)
    weights = 1.0 - weights / weights.sum()
    weights /= weights.min()
    return weights

def get_fc_fer_weights(num_classes):
    weights = {
        5: [378, 164, 622, 322, 632],
        6: [378, 504, 164, 622, 322, 632],
        7: [378, 104, 504, 164, 622, 322, 632],
    }
    return get_fc_weights(num_classes, weights.get(num_classes))

def get_cnn_fer_weights(num_classes):
    weights = {
        5: [5134, 5200, 9299, 6237, 4317],
        6: [5134, 799, 5200, 9299, 6237, 4317],
        7: [5134, 52, 799, 5200, 9299, 6237, 4317],
    }
    return get_fc_weights(num_classes, weights.get(num_classes))

def get_face_expression(inputs, predictor):
    predictions = predictor(inputs).detach()
    probalities = F.softmax(predictions, dim=1)
    print(probalities)
    maxim = probalities.max(axis=1)
    zeros = torch.zeros((len(predictions), 1))
    probalities = torch.cat([zeros, probalities], dim=1)
    probalities[:, 0] = 1 - maxim.values
    outputs = maxim.indices + 1
    outputs[maxim.values <= 0.7] = 0
    names = get_fer_names(predictor.num_classes)
    names = np.array(names, dtype=np.str)
    names = np.insert(names, 0, 'neutral')
    items = dict(zip(names, probalities.T))
    items['output'] = names[outputs.tolist()]
    return items

def get_face_features(image, detector, predictor2d, predictor3d, transform):
    rectangles = get_faces_rectangles(image, detector, 1)
    if len(rectangles) == 0:
        return False, [], [], []
    landmarks = get_face_landmarks_2d(image, rectangles, predictor2d)
    facemarks = get_facemarks_3d(landmarks, transform, predictor3d)
    return True, rectangles, landmarks, facemarks
