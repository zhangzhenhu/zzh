import collections

DataSet = collections.namedtuple('DataSet', ['name', 'x', 'y', 'predict', 'threshold'])

# class DataSet:
#
#     def __init__(self):
#         self.train_x = None
#         self.train_y = None
#         self.test_x = None
#         self.test_y = None
#         self.header = None
