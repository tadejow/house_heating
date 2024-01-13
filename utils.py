import matplotlib.pyplot as plt
import numpy as np
import cloudpickle
import base64
from PIL import Image


def lambda2str(expr):
    """
    :param expr:
    :return:
    """
    b = cloudpickle.dumps(expr)
    s = base64.b64encode(b).decode()
    return s


def str2lambda(s):
    """
    :param s:
    :return:
    """
    b = base64.b64decode(s)
    expr = cloudpickle.loads(b)
    return expr


def grayscale_array_to_coolwarm_image(matrix):
    """
    :param matrix:
    :return:
    """
    cm = plt.get_cmap("coolwarm")
    image = Image.fromarray(np.uint8(cm(matrix / np.max(matrix))[:, :, :3] * 255))
    rgb_image = Image.new("RGBA", image.size)
    rgb_image.paste(image)
    return rgb_image


def single_timestep_in_evolution(matrix, dt, dx, diffusion_coefficient, force_term):
    """
    :param matrix:
    :param dt:
    :param dx:
    :param diffusion_coefficient:
    :param force_term:
    :return:
    """
    # diffusion effect
    matrix[1:-1, 1:-1] += dt * diffusion_coefficient * (
            matrix[:-2, 1:-1] + matrix[2:, 1:-1] + matrix[1:-1, :-2] + matrix[1:-1, 2:] - 4 * matrix[1:-1, 1:-1]
    ) / (dx ** 2)
    # force term effect
    matrix += dt * force_term
    # boundaries
    matrix[0, :] = matrix[1, :]
    matrix[-1, :] = matrix[-2, :]
    matrix[:, 0] = matrix[:, 1]
    matrix[:, -1] = matrix[:, -2]
    return matrix


def single_timestep_matrix_merge(matrix, axis):
    """
    :param matrix:
    :param axis:
    :return:
    """
    return np.sum(matrix, axis)
