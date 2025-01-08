
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



class GMM_fitting():
    def __init__(self, n_components=3, covariance_type='full', random_state=42, rewardfn_path = "rewardfns/2d_gmm.pkl"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
    
        if not os.path.exists(rewardfn_path):

            gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
            np.random.seed(42)
            data1 = np.random.randn(300, 2) * 1.3 - 4      # 원래 데이터 (평균 대략 (-4,-4))
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


class GMM:
    def __init__(
        self,
        weights=[0.5, 0.5],         # shape = (K,)
        means=[[-4.0, -4.0],[4.0, 4.0]],           # shape = (K, 2)
        covariance=1.0,     # shape = (K, 2, 2)
        random_state=42,
        model_path="rewardfns/gmm_model.pkl",
        load_model=False
    ):
        """
        Initialize GMM model.

        Parameters:
        -----------
        weights : np.ndarray or list
            mixing coefficients of GMM, shape = (K,), sum(weights) must be 1
        means : np.ndarray
            means of GMM components, shape = (K, 2)
        covariance : np.ndarray
            covariance of GMM components, shape = (K, 2, 2)
        model_path : str
            path to save/load the model parameters
        random_state : int
            seed for random number generator
        load_model : bool
            if True, load the model parameters from the model_path, ignoring other parameters
        """
        np.random.seed(random_state)
        self.model_path = model_path
        K = len(weights)
        assert K == len(means) == len(weights), "Number of components must be the same for weights and means"

        if load_model:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    params = pickle.load(f)
                self.weights = params['weights']
                self.means = params['means']
                self.covariance = params['covariance']
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        else:
            assert sum(weights) == 1.0, "Sum of weights must be 1"

            self.weights = np.array(weights)
            self.means = np.array(means)
            self.covariance = np.array([
                [[covariance, 0.0], [0.0, covariance]] for _ in range(K)
            ], dtype=float)

            params = {
                'weights': self.weights,
                'means': self.means,
                'covariance': self.covariance
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(params, f)


        self._plot_model()

    def _multivariate_gaussian_pdf(self, x, mean, cov):
        """
        calculate pdf of multivariate gaussian distribution
        x:     (2,) shape numpy array
        mean:  (2,) shape
        cov:   (2, 2) shape
        """
        D = len(mean)  # 2차원
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        # 정규화 상수
        norm_factor = 1.0 / (np.sqrt((2 * np.pi) ** D * cov_det))

        diff = x - mean
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)

        return norm_factor * np.exp(exponent)

    def _gmm_pdf(self, x):
        """
        return pdf value of GMM at x
        x: (2,) shape numpy array
        """
        pdf_val = 0.0
        for w, m, c in zip(self.weights, self.means, self.covariance):
            pdf_val += w * self._multivariate_gaussian_pdf(x, m, c)
        return pdf_val

    def _plot_model(self, xlim=(-8, 8), ylim=(-8, 8), num_grid=100):
        """
        visualize pdf of 2d GMM in countour
        """
        x_vals = np.linspace(xlim[0], xlim[1], num_grid)
        y_vals = np.linspace(ylim[0], ylim[1], num_grid)
        X, Y = np.meshgrid(x_vals, y_vals)

        pdf_grid = np.zeros_like(X)
        for i in range(num_grid):
            for j in range(num_grid):
                xy = np.array([X[i, j], Y[i, j]])
                pdf_grid[i, j] = self._gmm_pdf(xy)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, pdf_grid, levels=30, cmap='viridis')
        plt.colorbar(contour, label='PDF Value')
        plt.title("GMM Likelihood")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # mean of each component
        plt.scatter(self.means[:, 0], self.means[:, 1], c='red', marker='x', s=100, label='Centers')

        plt.legend()
        plt.savefig(self.model_path.replace('.pkl', '.png'), dpi=300)
        plt.show()

    def __call__(self, x):
        """
        calculate likelihood(= GMM pdf) of x

        Parameters:
        - x: Tensor of shape (..., 2)

        Returns:
        - Tensor of shape (...) (likelihoods)
        """

        x_np = x.detach().cpu().squeeze().numpy()

        # (N, 2)
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]

        pdf_list = []
        for xy in x_np:
            pdf_val = self._gmm_pdf(xy)  # GMM PDF
            pdf_list.append(pdf_val)

        pdf_tensor = torch.tensor(pdf_list, dtype=torch.float32, device=x.device)

        if pdf_tensor.shape[0] == 1:
            return pdf_tensor.squeeze()
        else:
            return pdf_tensor


class MultiCenterParaboloid:
    def __init__(
        self, 
        a=1.0, 
        b=1.0, 
        c=0.0, 
        sigma=1.0,
        centers=None,         # [(x0_1, y0_1), (x0_2, y0_2), ..., (x0_K, y0_K)]
        weights=None,         # [w1, w2, ..., wK], sum(weights)=1
        random_state=42,
        model_path="rewardfns/configs/multi_paraboloid.pkl",
        load_model=False
    ):
        """
        initialize a multi-center(k) paraboloid model. 

        f_k(x, y) = a*(x - x0_k)^2 + b*(y - y0_k)^2 + c

        Likelihood = sum_k [ w_k * exp( - f_k(x,y)^2 / (2 * sigma^2) ) ]

        Parameters:
        -----------
        a, b, c : float
            coefficients of the paraboloid.(all paraboloids share the same coefficients)
        sigma : float
            standard deviation of the Gaussian term.(all paraboloids share the same sigma)
        centers : list of tuple
            center of each paraboloid. ex: [(0,0), (2,2), ...].
            if None, initialize with a single center at (0,0).
        weights : list or np.ndarray
            weight of each paraboloid. sum(weights) must be 1.
            if None, initialize with uniform distribution.
        random_state : int
            seed for random number generator.
        model_path : str
            path to save the model parameters.
        load_model : bool
            if True, load the model parameters from the model_path, ignore other parameters.
        """
        self.a = a
        self.b = b
        self.c = c
        self.sigma = sigma
        self.random_state = random_state
        self.model_path = model_path


        if centers is None or len(centers) == 0:
            centers = [(0.0, 0.0)]
        self.centers = np.array(centers, dtype=np.float32)  # shape = (K, 2)


        K = len(self.centers)
        if weights is None:
            weights = np.ones(K) / K
        else:
            weights = np.array(weights, dtype=np.float32)
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("sum of weights must be 1.")
        self.weights = weights

        assert len(self.centers) == len(self.weights), "number of centers and weights must be the same."

        if load_model:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    params = pickle.load(f)
                    self.a = params['a']
                    self.b = params['b']
                    self.c = params['c']
                    self.sigma = params['sigma']
                    self.centers = params['centers']
                    self.weights = params['weights']
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        else:
            params = {
                'a': self.a,
                'b': self.b,
                'c': self.c,
                'sigma': self.sigma,
                'centers': self.centers,
                'weights': self.weights
            }
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(params, f)

        self._plot_model()

    def _plot_model(self, xlim=(-8, 8), ylim=(-8, 8), num_grid=100):
        """
        plot the likelihood surface of the multi-center paraboloid model.
        """
        np.random.seed(self.random_state)

        x_vals = np.linspace(xlim[0], xlim[1], num_grid)
        y_vals = np.linspace(ylim[0], ylim[1], num_grid)
        X, Y = np.meshgrid(x_vals, y_vals)

        pdf_grid = np.zeros_like(X, dtype=np.float32)

        for i in range(num_grid):
            for j in range(num_grid):
                x_ij = X[i, j]
                y_ij = Y[i, j]
                pdf_val = 0.0
                for w_k, (cx, cy) in zip(self.weights, self.centers):
                    f_k = self.a * (x_ij - cx)**2 + self.b * (y_ij - cy)**2 + self.c
                    pdf_val += w_k * np.exp(- (f_k ** 2) / (2.0 * self.sigma ** 2))
                pdf_grid[i, j] = pdf_val

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, pdf_grid, levels=40, cmap='viridis')
        plt.colorbar(contour, label='Likelihood')

        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='red', marker='x', s=100, label='Centers')

        plt.title("Multi-Center Paraboloid Likelihood")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.savefig(self.model_path.replace('.pkl', '.png'), dpi=300)
        plt.show()

    def __call__(self, x):
        """
        calculate likelihood for each sample in x
        Parameters:
        -----------
        x : torch.Tensor of shape (..., 2)

        Returns:
        --------
        torch.Tensor of shape (...):
            likelihood for each sample(weighted sum of K paraboloids)  
        """
        # x: (N, 2) or (2,) 
        x_np = x.detach().cpu().squeeze().numpy()

        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)  # (1, 2)

        N = x_np.shape[0]
        K = len(self.centers)

        # 결과 배열 (N,)
        likelihoods = np.zeros(N, dtype=np.float32)

        for i in range(N):
            xi, yi = x_np[i]
            pdf_val = 0.0
            for w_k, (cx, cy) in zip(self.weights, self.centers):
                f_k = self.a * (xi - cx)**2 + self.b * (yi - cy)**2 + self.c
                pdf_val += w_k * np.exp(- (f_k ** 2) / (2.0 * self.sigma ** 2))
            likelihoods[i] = pdf_val

        return torch.tensor(likelihoods, dtype=torch.float32, device=x.device)