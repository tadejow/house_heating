import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def grayscale_array_to_coolwarm_image(matrix):
    cm = plt.get_cmap("coolwarm")
    image = Image.fromarray(np.uint8(cm(matrix / np.max(matrix))[:, :, :3] * 255))
    rgb_image = Image.new("RGBA", image.size)
    rgb_image.paste(image)
    return rgb_image
