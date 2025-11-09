# House Heating Simulation

## Project Description

This project utilizes the heat equation to simulate and analyze the thermal dynamics within a house layout. It provides a framework for modeling temperature distribution across different areas, including rooms, walls, windows, and doors, and simulates the effect of heat sources like radiators. The core of the project is the `HeatingModel` class, which uses a finite difference method to solve the heat equation over a discretized grid representing the house.

## Project Structure

```
.
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── model_params.json
└── pipeline/
    ├── main.py
    ├── utils.py
    └── experiments.py
```

## How to Run the Simulation

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tadejow/house_heating.git
    cd house_heating
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

The simulation is controlled by a set of parameters defined in a Python dictionary. You can run experiments by creating a script (e.g., `pipeline/experiments.py`) that imports the `HeatingModel`, defines the parameters, and runs the simulation.

#### 1. Configure Model Parameters

The model's behavior is defined by a single dictionary. This dictionary describes the house layout, physical properties, and initial conditions.

*Example `model_parameters` structure:*```python
model_parameters = {
    "areas": {
        "A1": {
            "row_min": 0, "row_max": 33, "col_min": 0, "col_max": 50,
            "init_func": lambda x: 285 + 5 * np.random.random(x.shape),
            "desired_temp": 300
        },
        # ... other area definitions ...
    },
    "walls": {
        # ... wall definitions ...
    },
    "doors": {
        # ... door definitions ...
    },
    "windows": {
        # ... window definitions ...
    },
    "radiators": {
        # ... radiator definitions ...
    },
    "domain": {
        "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0], "dx": 1
    },
    "force_term": lambda x, t, mask: np.where(mask == 1, 1.5, ...),
    "window_temp": lambda t: 270 - 5 * np.sin(24 * t / 3600),
    "diffusion_coefficient": 0.5,
    "current_time": 0.0
}```

**Parameter Descriptions:**
*   `areas`: Defines the rooms. Each area requires:
    *   `row_min`, `row_max`, `col_min`, `col_max`: The grid boundaries of the area.
    *   `init_func`: A lambda function to set the initial temperature.
    *   `desired_temp`: The target temperature (in Kelvin) for the room's thermostat.
*   `walls`, `doors`, `windows`, `radiators`: Dictionaries defining the grid coordinates for each of these elements.
*   `domain`: Contains the grid information. `dx` is the spatial step size.
*   `force_term`: A lambda function that defines the heat output of the radiators based on the radiator mask.
*   `window_temp`: A lambda function that defines the external temperature at the windows over time `t`.
*   `diffusion_coefficient`: The thermal diffusivity constant (`α`) for the heat equation.
*   `current_time`: The starting time for the simulation.

#### 2. Run an Experiment Script

Create a Python script (e.g., `pipeline/experiments.py`) to run the simulation.

*Example `experiments.py`:*
```python
import numpy as np
from pipeline.main import HeatingModel

# Define the model_parameters dictionary as shown above
# Or, load it from a JSON file:
# model = HeatingModel({}).load_params_from_file("data/model_params.json")

model_parameters = {
    # ... Paste the full parameter dictionary here ...
}

# 1. Initialize the model
model = HeatingModel(model_parameters)

# 2. Evolve the simulation for a number of steps
# E.g., 5000 steps with a time-step of 1.0
model.evolve(n_steps=5000, dt=1.0)

# 3. Generate and display an image of the final state
final_frame = model.build_image_frame()
final_frame.show()

# To save the frame:
# final_frame.save("results/final_temperature_distribution.png")```

To run the script, execute the following command in your terminal:
```bash
python pipeline/experiments.py```
The script will initialize the model, run the simulation, and then display an image representing the final temperature distribution.

### Visualizing the House Layout

The `build_image_frame()` method generates an image of the house, where colors represent different elements:

*   **Walls:** Black
*   **Windows:** Blue
*   **Doors:** Brown
*   **Rooms (Areas):** Colored based on temperature using a cool-to-warm colormap.

*(image of the house layout)*
