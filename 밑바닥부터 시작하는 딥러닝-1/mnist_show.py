import sys, os

from fontTools.ttLib.tables.T_S_I__0 import tsi0Format

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

img = x_train[0]
label = t_train[0]


img = img.reshape(28, 28)


img_show(img)