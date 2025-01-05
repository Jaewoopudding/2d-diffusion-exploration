import torch
import torch.nn as nn
import numpy as np
import functools
import copy
import os
import torch.nn.functional as F
import tqdm
from scipy.special import softmax
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import scipy
from tqdm import tqdm

# Dataset iterator
def inf_train_gen(data, batch_size=200, minimum=0):
    print(data)
    if data == "custom_density":
        scale = 1.
        # Define the unnormalized density function
        def unnormalized_density(x, y):
            return np.sin(10 * x) * np.cos(10 * y) + np.sin(5 * x * y) + 2.0 + minimum

        # Normalize the density function
        normalization_constant, _ = scipy.integrate.dblquad(
            lambda x, y: unnormalized_density(x, y),
            -1, 1,  # x bounds
            lambda x: -1, lambda x: 1  # y bounds
        )

        def normalized_density(x, y):
            return unnormalized_density(x, y) / normalization_constant

        # Sampling using rejection sampling
        def sample_from_density(n_samples):
            samples = []
            while len(samples) < n_samples:
                x_rand = np.random.uniform(-1, 1)
                y_rand = np.random.uniform(-1, 1)
                z_rand = np.random.uniform(0, 1)
                density_val = normalized_density(x_rand, y_rand)
                if z_rand < density_val:
                    samples.append((x_rand, y_rand))
            return np.array(samples) * scale

        samples = sample_from_density(batch_size)
        return samples.astype("float32"), None
    elif data == "swissroll":
        print(data)
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.sum(data**2, axis=-1,keepdims=True) / 9.0
    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data
    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        center_dist = X[:,0]**2 + X[:,1]**2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >=8.5)] = 0.667 
        energy[(center_dist >=5.0) & (center_dist <8.5)] = 0.333 
        energy[(center_dist >=2.0) & (center_dist <5.0)] = 1.0 
        energy[(center_dist <2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), energy[:,None]

    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data.astype(np.float32), (y > 0.5).astype(np.float32)[:,None]

    elif data == "8gaussians":
        scale = 1.
        centers = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                    (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        indexes = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.125
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            indexes.append(idx)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.array(indexes, dtype="float32")[:,None] / 7.0

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x, np.clip((1-np.concatenate([n,n]) / 10),0,1)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        points_x = points[:,0]
        judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))
        return points, judger.astype(np.float32)[:,None]

    elif data == "line":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    
    elif data == "cos":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    
    elif data == 'uniform':
        scale = 1.
        x = np.random.uniform(-1, 1, batch_size)
        y = np.random.uniform(-1, 1, batch_size)
        return np.stack((x * scale, y * scale), 1), None
    
    else:
        assert False

class Toy_dataset(torch.utils.data.Dataset):
    def __init__(self, name, datanum=1000000, minimum=0, device='cpu'):
        assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals", "uniform", "custom_density"]
        self.datanum =datanum
        self.name = name
        self.datas, _ = inf_train_gen(name, batch_size=datanum)
        self.datas = torch.Tensor(self.datas).to(device)
        self.datadim = 2

    def __getitem__(self, index):
        return self.datas[index]

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.datanum
    
    def visualize_and_save(self, batch_size):
        import matplotlib.pyplot as plt
        data_to_plot = self.datas[:batch_size].cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], s=1, alpha=0.2)
        plt.title(f'Dataset Visualization: {self.name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.savefig(f'images/{self.name}_visualization.png')
        plt.show()
