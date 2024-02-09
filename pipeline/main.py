import json
import tqdm
import utils
import numpy as np
import matplotlib.pyplot as plt


class HeatingModel:

    def __init__(self, params: dict):
        """
        :param params:
        """
        # load the model parameters
        self.params = params
        # initialize the part domains
        self.partial_matrix = {}
        # initialize the full domain
        self.result_matrix = np.zeros((100, 100))
        # initialize the mask matrix
        self.mask_matrix = np.zeros((100, 100))
        # build full and partial matrices and fill them with initial data
        self.build_partial_matrix()
        self.build_result_matrix()
        self.set_initial_data()
        # initialize the energy usage list
        self.energy_usage = []

    def load_params_from_file(self, fp: str):
        """
        :param fp: str - filepath
        :return: loads parameters of the model from the given filepath
        """
        self.params = json.loads(fp)
        self.params["force_term"] = utils.str2lambda(self.params["force_term"])
        self.params["window_temp"] = utils.str2lambda(self.params["window_temp"])
        for key in self.params["areas"].keys():
            self.params["areas"][key]["init_func"] = utils.str2lambda(self.params["areas"][key]["init_func"])
        return self

    def save_params_to_file(self, fp: str):
        """
        :param fp: str - filepath
        :return: saves parameters of the model to the given filepath
        """
        self.params["force_term"] = utils.lambda2str(self.params["force_term"])
        self.params["window_temp"] = utils.lambda2str(self.params["window_temp"])
        self.params["domain"]["a"], self.params["domain"]["b"], self.params["domain"]["N"] = \
            np.min(self.params["domain"]["grid"]), np.max(self.params["domain"]["grid"]), \
            self.params["domain"]["grid"].shape[0]
        self.params["domain"]["grid"] = "placeholder"
        for key in self.params["areas"].keys():
            self.params["areas"][key]["init_func"] = utils.lambda2str(self.params["areas"][key]["init_func"])
        with open(fp, "w") as outfile:
            json.dump(self.params, outfile)
        return self

    def build_partial_matrix(self):
        """
        :return: initializes / builds partial matrices (partitioned result matrix)
        """
        for key in self.params["areas"].keys():
            if key not in self.partial_matrix.keys():
                self.partial_matrix[key] = np.zeros(
                    (
                        self.params["areas"][key]["row_max"] - self.params["areas"][key]["row_min"],
                        self.params["areas"][key]["col_max"] - self.params["areas"][key]["col_min"]
                    )
                )
            else:
                self.partial_matrix[key] = self.result_matrix[
                    self.params["areas"][key]["row_min"]: self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]: self.params["areas"][key]["col_max"]
                ]
        return self

    def build_result_matrix(self):
        """
        :return: initializes mask matrix and builds the result matrix out of partial matrices
        """
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
        """
        :return: builds the PIL Image object out of result matrix and marks all inside elements
        """
        image = utils.grayscale_array_to_coolwarm_image(self.result_matrix)
        for key in self.params["walls"].keys():
            for p1 in range(self.params["walls"][key]["row_min"], self.params["walls"][key]["row_max"]):
                for p2 in range(self.params["walls"][key]["col_min"], self.params["walls"][key]["col_max"]):
                    image.putpixel((p2, p1), (0, 0, 0))
        for key in self.params["windows"].keys():
            for p1 in range(self.params["windows"][key]["row_min"], self.params["windows"][key]["row_max"]):
                for p2 in range(self.params["windows"][key]["col_min"], self.params["windows"][key]["col_max"]):
                    image.putpixel((p2, p1), (0, 0, 255))
        for key in self.params["doors"].keys():
            for p1 in range(self.params["doors"][key]["row_min"], self.params["doors"][key]["row_max"]):
                for p2 in range(self.params["doors"][key]["col_min"], self.params["doors"][key]["col_max"]):
                    image.putpixel((p2, p1), (150, 75, 0))
        return image

    def set_initial_data(self):
        """
        :return: evaluates the initial function on all partial matrices then updating the result matrix
        """
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
        """
        :param dt: float - timestep
        :return: performs one iteration of finite difference scheme for solving the heat equation
        """
        force_term_full = self.params["force_term"](self.params["domain"]["grid"],
                                                    self.params["current_time"],
                                                    self.mask_matrix)
        # firstly we set the temperature in the windows
        for key in self.params["windows"].keys():
            self.result_matrix[
                self.params["windows"][key]["row_min"]: self.params["windows"][key]["row_max"],
                self.params["windows"][key]["col_min"]: self.params["windows"][key]["col_max"]
            ] = self.params["window_temp"](self.params["current_time"])
        self.build_partial_matrix()
        # finally we solve the heat equation inside the rooms
        for key in self.params["areas"].keys():
            if self.partial_matrix[key].mean() >= self.params["areas"][key]["desired_temp"]:
                force_term_full[
                    self.params["areas"][key]["row_min"]: self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]: self.params["areas"][key]["col_max"]
                ] = 0
            else:
                pass
            self.partial_matrix[key] = utils.single_timestep_in_evolution(
                self.partial_matrix[key], dt, self.params["domain"]["dx"], self.params["diffusion_coefficient"],
                force_term_full[
                    self.params["areas"][key]["row_min"]: self.params["areas"][key]["row_max"],
                    self.params["areas"][key]["col_min"]: self.params["areas"][key]["col_max"]
                ]
            )
        self.build_result_matrix()
        # we allow the temperature exchange in the doors
        for key in self.params["doors"].keys():
            # horizontal doors
            if self.params["doors"][key]["row_max"] - self.params["doors"][key]["row_min"] == 2:
                self.result_matrix[
                    self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                    self.params["doors"][key]["col_min"]: self.params["doors"][key]["col_max"]
                ] = np.mean(self.result_matrix[
                        self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                        self.params["doors"][key]["col_min"]:self.params["doors"][key]["col_max"]
                    ], axis=0
                )
            # vertical doors
            else:
                self.result_matrix[
                    self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                    self.params["doors"][key]["col_min"]: self.params["doors"][key]["col_max"]
                ] = np.matrix(np.mean(self.result_matrix[
                        self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                        self.params["doors"][key]["col_min"]: self.params["doors"][key]["col_max"]
                    ], axis=1)
                ).T
        self.build_partial_matrix()
        self.energy_usage.append(force_term_full.sum())     # add energy usage for the timestep to the list
        self.params["current_time"] += dt   # update current time
        return self

    def evolve(self, n_steps: int, dt: float):
        """
        :param n_steps: int - number of timesteps
        :param dt: float - timestep
        :return: performs n steps of finite difference scheme for solving the heat equation
        """
        for _ in tqdm.tqdm(range(n_steps), desc="TIME STEPS"):
            self.evolve_in_unit_timestep(dt)
        self.energy_usage = np.cumsum(self.energy_usage)    # compute the cumulative sum of used energy
        return self

    def animate_evolution(self, n_steps: int, dt: float):

        return self

    def plot_evolution(self, n_steps: int, dt: float):

        return self


if __name__ == '__main__':
    model_parameters = {
        "areas": {
            "A1": {
                "row_min": 0, "row_max": 33, "col_min": 0, "col_max": 50,
                "init_func": lambda x: 295 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A2": {
                "row_min": 0, "row_max": 27, "col_min": 50, "col_max": 100,
                "init_func": lambda x: 298 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A3": {
                "row_min": 27, "row_max": 55, "col_min": 50, "col_max": 100,
                "init_func": lambda x: 297 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A4": {
                "row_min": 33, "row_max": 66, "col_min": 0, "col_max": 33,
                "init_func": lambda x: 296 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A5": {
                "row_min": 66, "row_max": 100, "col_min": 0, "col_max": 50,
                "init_func": lambda x: 298 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A6.1": {
                "row_min": 33, "row_max": 66, "col_min": 33, "col_max": 50,
                "init_func": lambda x: 295 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A6.2": {
                "row_min": 55, "row_max": 100, "col_min": 50, "col_max": 65,
                "init_func": lambda x: 295 + np.random.random(x.shape),
                "desired_temp": 300
            },
            "A7": {
                "row_min": 55, "row_max": 100, "col_min": 65, "col_max": 100,
                "init_func": lambda x: 288 + np.random.random(x.shape),
                "desired_temp": 300
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
                "row_min": 26, "row_max": 28, "col_min": 64, "col_max": 100
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
                "row_min": 55, "row_max": 100, "col_min": 64, "col_max": 66
            },
            "VV13": {
                "row_min": 54, "row_max": 56, "col_min": 64, "col_max": 100
            }
        },
        "doors": {
            "D1": {
                "row_min": 32, "row_max": 34, "col_min": 34, "col_max": 49
            },
            "D2": {
                "row_min": 26, "row_max": 28, "col_min": 51, "col_max": 66
            },
            "D3": {
                "row_min": 54, "row_max": 56, "col_min": 51, "col_max": 66
            },
            "D4": {
                "row_min": 65, "row_max": 67, "col_min": 34, "col_max": 49
            },
            "D5": {
                "row_min": 45, "row_max": 55, "col_min": 32, "col_max": 34
            },
            "D6": {
                "row_min": 55, "row_max": 65, "col_min": 49, "col_max": 51
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
                "row_min": 2, "row_max": 3, "col_min": 10, "col_max": 35, "mask_value": 1
            },
            "R2": {
                "row_min": 2, "row_max": 3, "col_min": 52, "col_max": 66, "mask_value": 2
            },
            "R3": {
                "row_min": 35, "row_max": 43, "col_min": 30, "col_max": 31, "mask_value": 3
            },
            "R4": {
                "row_min": 97, "row_max": 98, "col_min": 10, "col_max": 35, "mask_value": 4
            }
        },
        "domain": {
            "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0], "dx": 1
        },
        "force_term": lambda x, t, mask: np.where(
            mask == 1, 10**(-1), np.where(
                mask == 2, 10**(-1), np.where(
                    mask == 3, 10**(-1), np.where(
                        mask == 4, 10**(-1), 0
                    )
                )
            )
        ),
        "window_temp": lambda t: 280 - 5 * np.sin(24 * t / 3600),
        "diffusion_coefficient": 0.1,
        "current_time": 0.0

    }
    model = HeatingModel(model_parameters)
    # model.load_params_from_file("../data/params/params_v_01")
    plt.imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    plt.title(f"t = {model.params['current_time']}")
    plt.colorbar().set_label("Temperature [K]")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    model.evolve(1000, 0.5)
    model.result_matrix -= 273
    plt.imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    plt.title(f"t = {round(model.params['current_time'] / 3600, 1)} [h]")
    plt.colorbar().set_label("Temperature [C]")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    model.build_image_frame().resize((500, 500)).show()
    plt.show()

    
