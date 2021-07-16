import cv2
import numpy as np
import pandas as pd
from src.utils import DIR, IMG_SIZE

class FER2013(object):
    def __init__(self):
        self.class_names = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sadness',
            5: 'surprise',
            6: 'neutral',
        }

    def get_dataframe(self):
        csv_path = DIR.DATA.EXTERNAL.joinpath('FER2013.csv')
        df = pd.read_csv(csv_path)
        df.emotion = df.emotion.replace(self.class_names)
        return df

    def __iter__(self):
        df = self.get_dataframe()
        for i, row in df.iterrows():
            class_name = row['emotion']
            content = row.pixels.split(' ')
            image = np.array(content, dtype=np.uint8)
            if image.sum() == 0: continue
            image = image.reshape((48, 48))
            # image = cv2.resize(image, IMG_SIZE)
            yield image, class_name

if __name__ == '__main__':

    dataset = FER2013()

    for data in dataset:
        print(data)
