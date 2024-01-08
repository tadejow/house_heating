import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def grayscale_array_to_coolwarm_image(matrix):
    cm = plt.get_cmap("coolwarm")
    image = Image.fromarray(np.uint8(cm(matrix / np.max(matrix))[:, :, :3] * 255))
    rgb_image = Image.new("RGBA", image.size)
    rgb_image.paste(image)
    return rgb_image


def single_timestep_in_evolution(matrix, dt, dx, diffusion_coefficient, force_term):
    # diffusion effect
    matrix[1:-1, 1:-1] = dt * diffusion_coefficient * (
            matrix[:-2, 1:-1] + matrix[2:, 1:-1] + matrix[1:-1, :-2] + matrix[1:-1, 2:] - 4 * matrix[1:-1, 1:-1]
    ) / dx ** 2
    # force term effect
    matrix += dt * force_term
    # boundaries
    matrix[0, :] = matrix[1, :]
    matrix[-1, :] = matrix[-2, :]
    matrix[:, 0] = matrix[:, 1]
    matrix[:, -1] = matrix[:, -2]
    return matrix

def single_timestep_matrix_merge(matrix, axis):
    return np.sum(matrix, axis)
