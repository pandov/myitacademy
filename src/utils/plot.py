import cv2
import numpy as np
import matplotlib
from .connections import connections
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Plot(object):

    def __init__(self, projection: str = '2d', figsize: tuple = None):
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvasAgg(self.fig)
        self.projection = projection.lower()
        if self.projection == '2d':
            self.ax = self.fig.gca()
        elif self.projection == '3d':
            self.ax = Axes3D(self.fig)
        self.clear()

    def rotate_axes(self, elev: int = None, azim: int = None):
        if self.projection == '3d':
            self.ax.view_init(elev, azim)

    def clear(self):
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if self.projection == '3d':
            self.ax.set_zlabel('Z')

    def imshow(self, winname: str, waitable: bool = False):
        cv2.imshow(winname, self.to_image())
        if waitable and cv2.waitKey(0) == 27:
            cv2.destroyWindow(winname)

    def imwrite(self, filename: str, params: dict = None):
        cv2.imwrite(filename, self.to_image(), params)

    def to_image(self):
        self.ax.margins(0)
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
        w, h = self.canvas.get_width_height()
        image = buf.reshape((h, w, 4))[:, :, 1:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

class FacemarksPlot(Plot):

    def __init__(self):
        super().__init__(projection='3D', figsize=(10, 6))
        # self.show_trackbars()

    # def show_trackbars(self):
    #     na = lambda _: None
    #     cv2.namedWindow('Mask')
    #     cv2.createTrackbar('Eval', 'Mask', 45, 180, na)
    #     cv2.createTrackbar('Azim', 'Mask', 45, 180, na)

    def show2d(self, frame, is_detected, pts2d, mesh=True):
        if is_detected and mesh:
            pts2d = pts2d[0]
            for i, j in connections:
                cv2.line(frame, tuple(pts2d[i]), tuple(pts2d[j]), (0, 255, 0), 1)
        cv2.imshow('2D Mask', frame)

    def show3d(self, is_detected, pts3d):
        self.clear()
        if is_detected:
            pts3d = pts3d[0]
            for i, j in connections:
                p, q = pts3d[i], pts3d[j]
                line = Line3D((p[0], q[0]), (p[1], q[1]), (p[2], q[2]), c='green')
                self.ax.add_line(line)
            self.ax.scatter(*pts3d.T, c='black', linewidths=0.4, alpha=0.4)
            elev = cv2.getTrackbarPos('Eval', 'Mask')
            azim = cv2.getTrackbarPos('Azim', 'Mask')
            self.rotate_axes(elev, azim)
        self.imshow('3D Mask')

class EmotionPlot(Plot):

    def __init__(self):
        super().__init__(projection='2D', figsize=(8, 4))

    def show_bar(self, is_detected, expressions):
        self.clear()
        if is_detected:
            expressions = expressions.copy()
            del expressions['output']
            keys = list(expressions.keys())
            values = np.array(list(expressions.values())).T
            self.ax.bar(keys, values[0], align='center', alpha=0.6)
        self.imshow('Emotions')
