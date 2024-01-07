import itertools

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class HeatingModel:
    def __init__(self, params):
        # load the model parameters
        self.params = params
        # initialize the mini domains
        self.partial_matrix = {}
        # initialize the maxi domain
        self.result_matrix = np.zeros((100, 100))

    def build_partial_matrix(self):
        for key in self.params["areas"].keys():
            self.partial_matrix[key] = np.zeros(
                (
                    self.params["areas"][key]["row_max"] - self.params["areas"][key]["row_min"],
                    self.params["areas"][key]["col_max"] - self.params["areas"][key]["col_min"]
                )
            )
            self.partial_matrix[key] = self.params["areas"][key]["mask"]
        return self

    def set_initial_data(self):
        for key in self.params["areas"].keys():
            self.partial_matrix[key] = self.params["areas"][key]["init_func"](
                self.params["domain"][
                    self.params["areas"][key]["row_min"]:self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]:self.params["areas"][key]["col_max"]
                ]
            )
        return self

    def build_result_matrix(self):
        for key in self.params["areas"].keys():
            self.result_matrix[
                self.params["areas"][key]["row_min"]:self.params["areas"][key]["row_max"],
                self.params["areas"][key]["col_min"]:self.params["areas"][key]["col_max"]
            ] = self.partial_matrix[key]
        return self

    def build_image_frame(self):
        image = Image.fromarray(np.uint8(self.result_matrix))
        for key in self.params["walls"].keys():
            for p1 in range(self.params["walls"][key]["row_min"], self.params["walls"][key]["row_max"]):
                for p2 in range(self.params["walls"][key]["col_min"], self.params["walls"][key]["col_max"]):
                    image.putpixel((p1, p2), 0)
        return image


if __name__ == '__main__':
    model_parameters = {
        "areas": {
            "A1": {
                "row_min": 0, "row_max": 35, "col_min": 0, "col_max": 55, "mask": 1, "init_func": lambda x: 100
            },
            "A2": {
                "row_min": 0, "row_max": 35, "col_min": 55, "col_max": 100, "mask": 0, "init_func": lambda x: 100
            },
            "A3": {
                "row_min": 35, "row_max": 65, "col_min": 0, "col_max": 55, "mask": 1, "init_func": lambda x: 200
            },
            "A4": {
                "row_min": 50, "row_max": 80, "col_min": 55, "col_max": 100, "mask": 1, "init_func": lambda x: 300
            },
            "A5": {
                "row_min": 65, "row_max": 100, "col_min": 0, "col_max": 55, "mask": 1, "init_func": lambda x: 400
            },
            "A6": {
                "row_min": 65, "row_max": 100, "col_min": 55, "col_max": 100, "mask": 1, "init_func": lambda x: 500
            }
        },
        "walls": {
            "VV1": {"row_min": 0, "row_max": 35, "col_min": 0, "col_max": 55},
            "VV2": {"row_min": 0, "row_max": 35, "col_min": 55, "col_max": 100},
            "VV3": {"row_min": 35, "row_max": 65, "col_min": 0, "col_max": 55},
            "VV4": {"row_min": 50, "row_max": 80, "col_min": 55, "col_max": 100},
            "VV5": {"row_min": 65, "row_max": 100, "col_min": 0, "col_max": 55},
            "VV6": {"row_min": 65, "row_max": 100, "col_min": 55, "col_max": 100}
        },
        "windows": {
            "W1": {"row_min": 0, "row_max": 35, "col_min": 0, "col_max": 55},
            "W2": {"row_min": 0, "row_max": 35, "col_min": 55, "col_max": 100},
            "W3": {"row_min": 35, "row_max": 65, "col_min": 0, "col_max": 55},
            "W4": {"row_min": 50, "row_max": 80, "col_min": 55, "col_max": 100},
            "W5": {"row_min": 65, "row_max": 100, "col_min": 0, "col_max": 55},
            "W6": {"row_min": 65, "row_max": 100, "col_min": 55, "col_max": 100}
        },
        "domain": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0]
    }
    model = HeatingModel(model_parameters)
    model.build_partial_matrix()
    model.set_initial_data()
    model.build_result_matrix()
    plt.imshow(model.result_matrix)
    plt.show()
    model.build_image_frame().show()
    plt.show()
