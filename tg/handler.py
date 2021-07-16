import cv2
import numpy as np
from io import BytesIO
from src.connections import connections
from src.dlib_wrapper import *
from src.utils import *
from pathlib import Path

# detector = dlib_cnn_face_detector()
detector = dlib_face_frontal_detector()
predictor2d = dlib_face_landmarks_predictor()
predictor3d = get_fm_depth_predictor()
landmarks_transform = get_landmarks_transform()
face_images_transform = get_face_images_transform()
num_classes = 5
predictor_fer_fc = get_fc_fer_predictor(num_classes=num_classes, hidden_size=284)
predictor_fer_cnn = get_cnn_fer_predictor(num_classes=num_classes)
predictor_fer_cascade = get_cascade_fer_predictor(num_classes=num_classes)
get_features = lambda image: get_face_features(image, detector, predictor2d, predictor3d, landmarks_transform)

CACHE_PATH = Path(__file__).absolute().parent.joinpath('cache')
CACHE_PATH.mkdir(exist_ok=True)

fer_colors = {
    'anger': (231, 30, 46),
    'contempt': (72, 82, 166),
    'disgust': (97, 45, 145),
    'fear': (25, 0, 0),
    'happy': (74, 200, 71),
    'sadness': (26, 97, 175),
    'surprise': (251, 233, 37),
    'neutral': (150, 150, 150),
}

def get_fer_color(name):
    color = fer_colors[name]
    r, g, b = color
    return (b, g, r)

def backup(message, image):
    savepath = CACHE_PATH.joinpath(str(message.chat.username))
    savepath.mkdir(exist_ok=True)
    savename = savepath.joinpath(str(message.date) + '.jpg')
    cv2.imwrite(savename.as_posix(), image)

def byte2numpy(byte):
    arr = np.frombuffer(byte, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image

def numpy2byte(image):
    _, data = cv2.imencode('.jpg', image)
    bio = BytesIO(data)
    bio.name = 'image.jpeg'
    bio.seek(0)
    return bio

def print_faces(image, rectangles, landmarks, expressions, mesh=False, rect=False):
    for i, rectangle in enumerate(rectangles):
        x, y, w, h = dlib_rectangle_to_numpy_box(image.shape, rectangle)
        output = expressions['output'][i]
        if mesh:
            color = get_fer_color(output)
            for j, k in connections:
                cv2.line(image, tuple(landmarks[i][j]), tuple(landmarks[i][k]), color, 1, cv2.LINE_AA)
        if rect:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)

        for n, (expression, values) in enumerate(expressions.items()):
            if expression == 'output': continue
            scale = 0.45
            if expression == output:
                expression = '*' + expression
                scale = 0.5
            value = float(values[i])
            text = '{} {}'.format(expression, round(value, 2))
            tx, ty = x + w // 5, y + n * 15 - h // 4
            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.rectangle(image, (tx, ty), (tx + w // 3, ty + h // 4 - 10), (0, 0, 0, 50), -1)
