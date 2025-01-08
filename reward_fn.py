
import torch
import numpy as np

import os
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture




def rewardfn(x):
    # Compute the distance from (0, 0)
    distances = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)  # Euclidean distance
    
    # Convert distances to proximity values
    proximity = 1 / (1 + distances)  # Avoid division by zero by adding 1

    return proximity



class GMM():
    def __init__(self, n_components=3, covariance_type='full', random_state=42, rewardfn_path = "rewardfns/2d_gmm.pkl"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
    
        if not os.path.exists(rewardfn_path):

            gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
            np.random.seed(42)
            data1 = np.random.randn(300, 2) * 1.3 - 4      # 원래 데이터 (평균 대략 (0,0))
            data2 = np.random.randn(300, 2) * 1.3 + 4      # (4,4) 부근 데이터 (평균 대략 (4,4))

            data = np.vstack([data1, data2])   
            gmm.fit(data)

            with open(rewardfn_path, "wb") as f:
                pickle.dump(gmm, f)
            
            self.gmm = gmm

        else:
            with open(rewardfn_path, "rb") as f:
                self.gmm = pickle.load(f)
                data = np.random.randn(300, 2) * 2

        # draw contour plot
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        XY = np.column_stack([X.ravel(), Y.ravel()])

        
        densities = np.exp(self.gmm.score_samples(XY))
        Z = densities.reshape(100, 100)

        
        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=10, cmap='viridis')
        plt.scatter(data[:, 0], data[:, 1], c='blue', s=10, alpha=0.5, label='Data points')
        plt.title("2D Gaussian Mixture Model")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        plt.savefig(f"rewardfns/2d_gmm.png", dpi=300)
        plt.show()


    def __call__(self, x):
        # Compute the log-likelihood of the input points
        log_likelihood = self.gmm.score_samples(x.detach().cpu().squeeze().numpy())
        return torch.exp(torch.tensor(log_likelihood, device=x.device))


