


import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import cv2
import os
import kaggle


import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb

import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


def parrot(aa, action='voom'):
    print(aa, 'and', end=' ')
    print(action)


from torchvision.models.resnet import resnet50

if __name__ == '__main__':
   labels=torch.Tensor([1,2,3]).long()
   print(labels)
   print(labels.unsqueeze(1))




