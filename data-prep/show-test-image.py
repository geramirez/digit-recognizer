import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show(d, index):

    img = d.iloc[index]
    if img.shape[0] == 785:
        img = img[1:]
    img = img.reshape(28,28)
    img = img.astype('uint8')
    plt.matshow(img)
