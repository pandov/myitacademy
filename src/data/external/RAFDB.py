import cv2
import numpy as np
from src.utils import DIR

class RAFDB(object):
    def __init__(self):
        self.class_names = {
            1: 'surprise',
            2: 'fear',
            3: 'disgust',
            4: 'happy',
            5: 'sadness',
            6: 'anger',
            7: 'neutral',
        }

    def get_image_class_map(self, filepath):
        with open(filepath, 'r') as f:
            lines = np.char.strip(f.readlines())
            lines = np.char.split(lines, sep=' ')
            lines = np.array(list(lines))
            pairs = dict(zip(lines[:, 0], lines[:, 1].astype(np.int32)))
            return pairs

    def __iter__(self):
        data_path = DIR.DATA.EXTERNAL.joinpath('RAF-DB')
        label_file = data_path.joinpath('EmoLabel/list_patition_label.txt')
        pairs = self.get_image_class_map(label_file)
        for filename, class_id in pairs.items():
            class_name = self.class_names[class_id]
            imagepath = data_path.joinpath('Image/original/' + filename).as_posix()
            image = cv2.imread(imagepath)
            yield image, class_name

if __name__ == '__main__':

    dataset = RAFDB()

    for data in dataset:
        print(data)
        break
