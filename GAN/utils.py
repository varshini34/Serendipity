import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random

def generate_random_vector(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0, size - 1)
    label_tensor[random_idx] = 1
    return label_tensor


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


def conto1(array, vector):
    q = np.zeros(3900)
    for i in range(len(array)):
        cosv = cosine_similarity(array[i].reshape(1, -1), vector.reshape(1, -1))
        q = q + cosv * array[i].reshape(-1, 1)  # Reshape array[i] to have shape (3900, 1)
    return minmaxscaler(q)


#assigning dates to generated rating within the timestamp
def change_date(a, epoch, path):
    b = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if (~np.isnan(a[i][j])):
                c = []
                c.append(i)
                c.append(j)
                c.append(a[i][j])
                c.append(random.randint(879000000, 892999999))
                b.append(c)
    df = pd.DataFrame(b)
    df.to_csv(path + '/data_df' + str(epoch) + '.csv', index=False, header=True)