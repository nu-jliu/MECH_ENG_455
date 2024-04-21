import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import requests

from PIL import Image
from io import BytesIO


IMAGE_URL = (
    "https://raw.githubusercontent.com/MurpheyLab/ME455_public/main/figs/lincoln.jpg"
)


def image_density(s, x_grid, y_grid, density_array):
    s_x, s_y = s

    ind_x = cp.argmin(cp.abs(x_grid - s_x))
    ind_y = cp.argmin(cp.abs(y_grid - s_y))

    val = density_array[ind_x, ind_y]

    return val


if __name__ == "__main__":
    response = requests.get(IMAGE_URL)
    image_data = BytesIO(response.content)

    image = Image.open(image_data)

    image_array = cp.array(image)
    image_array = cp.flipud(image_array)

    # print(image_array)

    plt.imshow(cp.asnumpy(image_array), origin="lower", cmap="gray")
    plt.show()
    plt.close()

    x_grids = cp.linspace(0.0, 1.0, image_array.shape[0])
    y_grids = cp.linspace(0.0, 1.0, image_array.shape[1])

    dx = x_grids[1]
    dy = y_grids[1]

    X, Y = cp.meshgrid(x_grids, y_grids)

    density_array = 255.0 - image_array
    density_array /= cp.sum(density_array) * dx * dy

    print(density_array)
    print(X, Y)

    samples = cp.random.uniform(low=0.0, high=1.0, size=(5000, 2))
    print(samples)

    sample_weights = cp.zeros(5000)
