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

if __name__ == '__main__':
    a=[]
    a.append(torch.tensor([1,2]))
    a.append(torch.tensor([1,3]))
    for aa in a:
        print(type(aa))


