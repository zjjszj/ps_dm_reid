import torch
import random

from torch import autograd
from datasets.process_ps_data import ps_data_manager


class Student(object):

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2014 - self._birth

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

if __name__ == '__main__':
    a=np.array([0.8,0.2,0.9])
    inds=np.argsort
    print(ap)




