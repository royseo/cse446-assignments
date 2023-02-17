from typing import List

import matplotlib.pyplot as plt
import numpy as np


from utils import problem

def  cdf_fucntion(z: np.ndarray, x: float = 1.0) -> float:
    return np.sum(z <= x) / z.shape[0]


def drow_cdf(z: np.ndarray, xs: float = 1.0):
    fn = []
    for x_i in xs.tolist():
        fn.append(cdf_fucntion(z, x_i))
    
    plt.step(xs, fn)


def drow_y_k(n: int, ks: List[int]):
    for k in ks:
        Y_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1.0 / k), axis=1)
        plt.step(sorted(Y_k), np.arange(1, n + 1) / float(n), label=str(k))

    # Plot gaussian
    Z = np.random.randn(n)
    plt.step(sorted(Z), np.arange(1, n + 1) / float(n), label="Gaussian")


def main():
    n = 20000
    z = np.random.randn(n)
    x = np.arange(-3, 3, 0.01)
    drow_cdf(z, x)
    plot_settings()
    
    required_std = 0.0025
    n = int(np.ceil(1.0 / (required_std * 2))) ** 2
    ks = [1, 8, 64, 512]
    drow_y_k(n, ks)
    plot_settings()


def plot_settings():
    # Plotting settings
    plt.grid(which="both", linestyle="dotted")
    plt.legend(
        loc="lower right", bbox_to_anchor=(1.1, 0.2), fancybox=True, shadow=True, ncol=1
    )
    plt.ylim((0, 1))
    plt.xlim((-3, 3))
    plt.ylabel("Probability")
    plt.xlabel("Observations")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
