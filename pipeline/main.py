import json
import tqdm
import utils
import numpy as np


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



    
