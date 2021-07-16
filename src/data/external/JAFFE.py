import cv2
import pandas as pd
from src.utils import DIR

class JAFFE(object):
    def __init__(self):
        self.column_renames = {
            'HAP': 'happy',
            'SAD': 'sadness',
            'SUR': 'surprise',
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'PIC': 'pattern',
        }

    def __iter__(self):
        data_path = DIR.DATA.EXTERNAL.joinpath('JAFFE')
        df = self.get_dataframe(data_path)
        columns = df.columns.values
        path_images = data_path.joinpath('Images')
        for index, row in df.iterrows():
            label = row[columns].values[:-1].argmax()
            class_name = columns[label]
            pattern = row['pattern']
            images_path = list(path_images.rglob(pattern))
            if len(images_path) == 0: continue
            imagepath = images_path[0].as_posix()
            image = cv2.imread(imagepath)
            yield image, class_name

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
        break
