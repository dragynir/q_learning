from typing import List

from matplotlib import pyplot as plt


def plot_values(
    x: List[float],
    y: List[float],
    title: str = 'values',
    xlabel: str = 'x',
    ylabel: str = 'y',
) -> None:
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
