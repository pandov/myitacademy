import cv2
import numpy as np
import pandas as pd
from ...utils import get_fer_names
from ...paths import iter_files, PATH_DATA_EXTERNAL

class JAFFE(object):

    def __init__(self, num_classes):
        self.column_renames = {
            'HAP': 'happy',
            'SAD': 'sadness',
            'SUR': 'surprise',
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'PIC': 'pattern',
        }
        self.include = get_fer_names(num_classes)

    def __iter__(self):
        path = PATH_DATA_EXTERNAL.joinpath('JAFFE')
        df = self.get_dataframe(path)
        columns = df.columns.values
        path_images = path.joinpath('Images')
        for index, row in df.iterrows():
            label = np.argmax(row[columns].values[:-1])
            class_name = columns[label]
            if class_name not in self.include: continue
            pattern = row['pattern']
            imagepaths = list(path_images.rglob(pattern))
            if len(imagepaths) == 0: continue
            imagefile = imagepaths[0].as_posix()
            image_gray = self.to_grayscale(imagefile)
            yield image_gray, class_name

    def to_grayscale(self, imagefile):
        image = cv2.imread(imagefile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_dataframe(self, path):
        csv_with_fear = path.joinpath('SRD.csv')
        csv_without_fear = path.joinpath('SRD_FEAR_EXCLUDE.csv')
        kwargs = {'sep': ' ', 'index_col': '#'}
        df_with_fear = pd.read_csv(csv_with_fear, **kwargs)
        df_without_fear = pd.read_csv(csv_without_fear, **kwargs)
        df_without_fear['FEA'] = 0.0
        df = pd.concat((df_with_fear, df_without_fear), ignore_index=True)
        df['PIC'] = df['PIC'].str.replace('-','.') + '.*.tiff'
        df = df.rename(columns=self.column_renames)
        return df

if __name__ == '__main__':

    dataset = JAFFE()

    for data in dataset:
        print(data)
