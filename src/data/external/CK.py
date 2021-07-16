import cv2
from src.utils import DIR

class CK(object):
    def __init__(self):
        self.class_names = {
            0: 'neutral',
            1: 'anger',
            2: 'contempt',
            3: 'disgust',
            4: 'fear',
            5: 'happy',
            6: 'sadness',
            7: 'surprise',
        }

    def get_class_name(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read().strip()
            label = int(round(float(content), 2))
            return self.class_names[label]

    def __iter__(self):
        data_path = DIR.DATA.EXTERNAL
        for filepath in data_path.joinpath('CK/Emotions').rglob('*.txt'):
            class_name = self.get_class_name(filepath)
            images_path = data_path / str(filepath.parent).replace('Emotions', 'Images')
            images_path = list(images_path.rglob('*.png'))
            amount = 2 if len(images_path) < 9 else 3
            for i in range(2):
                imagepath = images_path[-(1 + i * 2)].as_posix()
                image = cv2.imread(imagepath)
                yield image, class_name

if __name__ == '__main__':

    dataset = CK()

    for data in dataset:
        print(data)
        break
