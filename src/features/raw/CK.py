import cv2
from ...utils import get_fer_names
from ...paths import iter_files, PATH_DATA_EXTERNAL

class CK(object):

    def __init__(self, num_classes):
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
        self.include = get_fer_names(num_classes)

    def __iter__(self):
        path_emotions = PATH_DATA_EXTERNAL.joinpath('CK').joinpath('Emotions')
        for filepath in iter_files(path_emotions):
            class_name = self.get_class_name(filepath)
            if class_name not in self.include: continue
            imagedir = str(filepath.parent).replace('Emotions', 'Images')
            imagepath = PATH_DATA_EXTERNAL / imagedir
            imagefiles = list(iter_files(imagepath))
            amount = 2 if len(imagefiles) < 9 else 3
            for i in range(amount):
                imagefile = imagefiles[-amount].as_posix()
                image_gray = self.to_grayscale(imagefile)
                yield image_gray, class_name

    def to_grayscale(self, imagefile):
        image = cv2.imread(imagefile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_class_name(self, filepath):
        with open(filepath, 'r') as f:
            content = f.read().strip()
            label = int(round(float(content), 2))
            return self.class_names[label]

if __name__ == '__main__':

    dataset = CK()

    for data in dataset:
        print(data)
