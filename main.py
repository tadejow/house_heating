import tqdm
import utils
import numpy as np
import matplotlib.pyplot as plt


class HeatingModel:

    def __init__(self, params: dict):
        # load the model parameters
        self.params = params
        # initialize the part domains
        self.partial_matrix = {}
        # initialize the full domain
        self.result_matrix = np.zeros((100, 100))
        # initialize the mask matrix
        self.mask_matrix = np.zeros((100, 100))

    def build_partial_matrix(self):
        for key in self.params["areas"].keys():
            self.partial_matrix[key] = np.zeros(
                (
                    self.params["areas"][key]["row_max"] - self.params["areas"][key]["row_min"],
                    self.params["areas"][key]["col_max"] - self.params["areas"][key]["col_min"]
                )
            )
        return self

    def build_result_matrix(self):
        for key in self.params["areas"].keys():
            self.result_matrix[
                self.params["areas"][key]["row_min"]:self.params["areas"][key]["row_max"],
                self.params["areas"][key]["col_min"]:self.params["areas"][key]["col_max"]
            ] = self.partial_matrix[key]
        if self.mask_matrix.sum() == 0.0:
            for key in self.params["radiators"].keys():
                self.mask_matrix[
                    self.params["radiators"][key]["row_min"]: self.params["radiators"][key]["row_max"],
                    self.params["radiators"][key]["col_min"]: self.params["radiators"][key]["col_max"]
                ] = self.params["radiators"][key]["mask_value"]
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

    def set_initial_data(self):
        if "current_time" not in self.params.keys():
            self.params["current_time"] = 0.0
        for key in self.params["areas"].keys():
            self.partial_matrix[key] = self.params["areas"][key]["init_func"](
                self.params["domain"]["grid"][
                    self.params["areas"][key]["row_min"]:self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]:self.params["areas"][key]["col_max"]
                ]
            )
        self.build_result_matrix()
        return self

    def evolve_in_unit_timestep(self, dt: float):
        force_term_full = self.params["force_term"](self.params["domain"]["grid"],
                                                    self.params["current_time"],
                                                    self.mask_matrix)
        for key in self.params["areas"].keys():
            self.partial_matrix[key] = utils.single_timestep_in_evolution(
                self.partial_matrix[key], dt, self.params["domain"]["dx"], self.params["diffusion_coefficient"],
                force_term_full[
                    self.params["areas"][key]["row_min"]: self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]: self.params["areas"][key]["col_max"]
                ]
            )
        for key in self.params["walls"].keys():
            if self.params["walls"][key]["row_max"] - self.params["walls"][key]["row_min"] == 2:
                self.result_matrix[
                    self.params["walls"][key]["row_min"]: self.params["walls"][key]["row_max"],
                    self.params["walls"][key]["col_min"]: self.params["walls"][key]["col_max"]
                ] = np.mean(
                    self.result_matrix[
                        self.params["walls"][key]["row_min"]: self.params["walls"][key]["row_max"],
                        self.params["walls"][key]["col_min"]: self.params["walls"][key]["col_max"]
                    ], axis=0
                )
            else:
                self.result_matrix[
                    self.params["walls"][key]["row_min"]: self.params["walls"][key]["row_max"],
                    self.params["walls"][key]["col_min"]: self.params["walls"][key]["col_max"]
                ] = np.matrix(
                        np.mean(self.result_matrix[
                                    self.params["walls"][key]["row_min"]: self.params["walls"][key]["row_max"],
                                    self.params["walls"][key]["col_min"]: self.params["walls"][key]["col_max"]
                                ], axis=1)
                ).T
        return self

    def evolve(self, n_steps: int, dt: float):
        for _ in tqdm.tqdm(range(n_steps), desc="TIME STEPS"):
            self.evolve_in_unit_timestep(dt)
            self.params["current_time"] += dt
        self.build_result_matrix()
        return self


if __name__ == '__main__':
    model_parameters = {
        "areas": {
            "A1": {
                "row_min": 0, "row_max": 33, "col_min": 0, "col_max": 50,
                "init_func": lambda x: 285 + np.random.random(x.shape)
            },
            "A2": {
                "row_min": 0, "row_max": 27, "col_min": 50, "col_max": 100,
                "init_func": lambda x: 287 + np.random.random(x.shape)
            },
            "A3": {
                "row_min": 27, "row_max": 55, "col_min": 50, "col_max": 100,
                "init_func": lambda x: 287 + np.random.random(x.shape)
            },
            "A4": {
                "row_min": 33, "row_max": 66, "col_min": 0, "col_max": 33,
                "init_func": lambda x: 291 + np.random.random(x.shape)
            },
            "A5": {
                "row_min": 66, "row_max": 100, "col_min": 0, "col_max": 50,
                "init_func": lambda x: 288 + np.random.random(x.shape)
            },
            "A6.1": {
                "row_min": 33, "row_max": 66, "col_min": 33, "col_max": 50,
                "init_func": lambda x: 290 + np.random.random(x.shape)
            },
            "A6.2": {
                "row_min": 55, "row_max": 100, "col_min": 50, "col_max": 60,
                "init_func": lambda x: 290 + np.random.random(x.shape)
            },
            "A7": {
                "row_min": 55, "row_max": 100, "col_min": 60, "col_max": 100,
                "init_func": lambda x: 275 + np.random.random(x.shape)
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
        "radiators": {
            "R1": {
                "row_min": 2, "row_max": 3, "col_min": 17, "col_max": 33, "mask_value": 1
            },
            "R2": {
                "row_min": 2, "row_max": 3, "col_min": 52, "col_max": 66, "mask_value": 2
            },
            "R3": {
                "row_min": 35, "row_max": 43, "col_min": 30, "col_max": 31, "mask_value": 3
            },
            "R4": {
                "row_min": 97, "row_max": 98, "col_min": 17, "col_max": 33, "mask_value": 4
            }
        },
        "domain": {
            "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0], "dx": 1
        },
        "force_term": lambda x, t, mask: np.where(
            mask == 1, (np.cos(t)**2 + 1) * 10**(-1), np.where(
                mask == 2, (np.cos(t)**2 + 1) * 10**(-1), np.where(
                    mask == 3, (np.cos(t)**2 + 1) * 10**(-1), np.where(
                        mask == 4, (np.cos(t)**2 + 1) * 10**(-1), 0
                    )
                )
            )
        ),
        "diffusion_coefficient": 10**(-3)

    }
    model = HeatingModel(model_parameters)
    model.build_partial_matrix()
    model.set_initial_data()
    plt.imshow(model.result_matrix)
    plt.show()
    # model.build_image_frame().show()
    # plt.show()
    model.evolve(100000, 0.005)
    plt.imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    plt.colorbar()
    plt.show()
    # model.build_image_frame().show()
    # plt.show()
