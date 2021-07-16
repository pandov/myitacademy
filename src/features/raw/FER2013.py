import numpy as np
import pandas as pd
from ...utils import get_fer_names
from ...paths import PATH_DATA_EXTERNAL

class FER2013(object):

    def __init__(self, num_classes):
        self.class_names = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'sadness',
            5: 'surprise',
            6: 'neutral',
        }
        self.include = get_fer_names(num_classes)

    def __iter__(self):
        df = self.get_dataframe()
        for i, row in df.iterrows():
            class_name = row['emotion']
            if class_name not in self.include: continue
            content = row.pixels.split(' ')
            size = int(np.sqrt(len(content)))
            image = np.array(content, dtype=np.uint8)
            if image.sum() == 0: continue
            image = image.reshape((size, size))
            yield image, class_name

    def get_dataframe(self):
        csv_path = PATH_DATA_EXTERNAL.joinpath('FER2013.csv')
        df = pd.read_csv(csv_path)
        df.emotion = df.emotion.replace(self.class_names)
        return df

if __name__ == '__main__':

    dataset = FER2013()

    for data in dataset:
        print(data)
