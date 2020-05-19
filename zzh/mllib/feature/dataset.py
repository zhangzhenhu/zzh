import collections
import numpy as np


# DataSet = collections.namedtuple('DataSet', ['name', 'x', 'y', 'predict', 'threshold'],
#                                  defaults=["", None, None, None, 0.5])


class DataSet:

    def __init__(self, name="default",
                 x: np.ndarray = None,
                 y=None, predict=None, threshold=0.5,
                 header=[],
                 xi=None):
        self.x = x
        self.y = y
        self.predict = predict
        self.name = name
        self.threshold = threshold
        self.header = header
        self.xi = xi

    def other(self, other: "DataSet"):
        self.x = other.x
        self.y = other.y
        self.name = other.name
        self.predict = other.name
        self.threshold = other.threshold
        self.header = other.header
        self.xi = other.xi
        return self

    def update(self, name=None, x=None, y=None, predict=None, threshold=None, header=None, xi=None):
        if name is not None:
            self.name = name
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if predict is not None:
            self.predict = predict
        if threshold is not None:
            self.threshold = threshold

        if header is not None:
            self.header = header
        if xi is not None:
            self.xi = xi
        return self

    def set_threshold(self, threshold):
        self.threshold = threshold
        return self

    def copy(self):

        return DataSet().other(self)
# class DataSet:
#
#     def __init__(self):
#         self.train_x = None
#         self.train_y = None
#         self.test_x = None
#         self.test_y = None
#         self.header = None
