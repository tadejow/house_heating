import utils
import numpy as np
import matplotlib.pyplot as plt


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

    def evolve_in_unit_timestep(self, dt):
        # inside every area
        for key in self.params["areas"].keys():
            self.partial_matrix[key][1:-1, 1:-1] = self.partial_matrix[key][:-2, 1:-1]

    def build_result_matrix(self):
        for key in self.params["areas"].keys():
            self.result_matrix[
                self.params["areas"][key]["row_min"]:self.params["areas"][key]["row_max"],
                self.params["areas"][key]["col_min"]:self.params["areas"][key]["col_max"]
            ] = self.partial_matrix[key]
        return self

    def build_image_frame(self):
        image = utils.grayscale_array_to_coolwarm_image(self.result_matrix)
        for key in self.params["walls"].keys():
            for p1 in range(self.params["walls"][key]["row_min"], self.params["walls"][key]["row_max"]):
                for p2 in range(self.params["walls"][key]["col_min"], self.params["walls"][key]["col_max"]):
                    image.putpixel((p2, p1), (0, 0, 0))
        for key in self.params["windows"].keys():
            for p1 in range(self.params["windows"][key]["row_min"], self.params["windows"][key]["row_max"]):
                for p2 in range(self.params["windows"][key]["col_min"], self.params["windows"][key]["col_max"]):
                    image.putpixel((p2, p1), (0, 0, 255))
        return image


if __name__ == '__main__':
    model_parameters = {
        "areas": {
            "A1": {
                "row_min": 0, "row_max": 33, "col_min": 0, "col_max": 50, "mask": 1, "init_func": lambda x: -100
            },
            "A2": {
                "row_min": 0, "row_max": 27, "col_min": 50, "col_max": 100, "mask": 1, "init_func": lambda x: 200
            },
            "A3": {
                "row_min": 27, "row_max": 55, "col_min": 50, "col_max": 100, "mask": 1, "init_func": lambda x: 300
            },
            "A4": {
                "row_min": 33, "row_max": 66, "col_min": 0, "col_max": 33, "mask": 1, "init_func": lambda x: 400
            },
            "A5": {
                "row_min": 66, "row_max": 100, "col_min": 0, "col_max": 50, "mask": 1, "init_func": lambda x: 500
            },
            "A6.1": {
                "row_min": 33, "row_max": 66, "col_min": 33, "col_max": 50, "mask": 1, "init_func": lambda x: 600
            },
            "A6.2": {
                "row_min": 55, "row_max": 100, "col_min": 50, "col_max": 60, "mask": 1, "init_func": lambda x: 600
            },
            "A7": {
                "row_min": 55, "row_max": 100, "col_min": 60, "col_max": 100, "mask": 0, "init_func": lambda x: 100
            }
        },
        "walls": {
            "VV1": {
                "row_min": 0, "row_max": 100, "col_min": 0, "col_max": 1
            },
            "VV2": {
                 "row_min": 0, "row_max": 1, "col_min": 0, "col_max": 100
            },
            "VV3": {
                 "row_min": 0, "row_max": 100, "col_min": 99, "col_max": 100
            },
            "VV4": {
                "row_min": 99, "row_max": 100, "col_min": 0, "col_max": 100
            },
            "VV5": {
                "row_min": 0, "row_max": 55, "col_min": 49, "col_max": 51
            },
            "VV6": {
                "row_min": 26, "row_max": 28, "col_min": 59, "col_max": 100
            },
            "VV7": {
                "row_min": 32, "row_max": 34, "col_min": 0, "col_max": 34
            },
            "VV8": {
                "row_min": 33, "row_max": 45, "col_min": 32, "col_max": 34
            },
            "VV9": {
                "row_min": 55, "row_max": 66, "col_min": 32, "col_max": 34
            },
            "VV10": {
                "row_min": 65, "row_max": 67, "col_min": 0, "col_max": 34
            },
            "VV11": {
                "row_min": 65, "row_max": 100, "col_min": 49, "col_max": 51
            },
            "VV12": {
                "row_min": 55, "row_max": 100, "col_min": 59, "col_max": 61
            },
            "VV13": {
                "row_min": 54, "row_max": 56, "col_min": 59, "col_max": 100
            }
        },
        "windows": {
            "W1": {
                "row_min": 0, "row_max": 1, "col_min": 15, "col_max": 35
            },
            "W2": {
                "row_min": 0, "row_max": 1, "col_min": 70, "col_max": 90
            },
            "W3": {
                "row_min": 30, "row_max": 40, "col_min": 99, "col_max": 100
            },
            "W4": {
                "row_min": 99, "row_max": 100, "col_min": 15, "col_max": 35
            }
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
