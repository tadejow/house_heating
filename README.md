# \# House Heating Simulation

# 

# \## Project Description

# 

# This project utilizes the heat equation to simulate and analyze the thermal dynamics within a house layout. It provides a framework for modeling temperature distribution across different areas, including rooms, walls, windows, and doors, and simulates the effect of heat sources like radiators. The core of the project is the `HeatingModel` class, which uses a finite difference method to solve the heat equation over a discretized grid representing the house.

# 

# \## Project Structure

# 

# ```

# .

# ├── .gitignore

# ├── requirements.txt

# ├── README.md

# ├── data/

# │   └── model\_params.json

# └── pipeline/

# &nbsp;   ├── main.py

# &nbsp;   ├── utils.py

# &nbsp;   └── experiments.py

# ```

# 

# \## How to Run the Simulation

# 

# \### Installation

# 

# 1\.  \*\*Clone the repository:\*\*

# &nbsp;   ```bash

# &nbsp;   git clone https://github.com/tadejow/house\_heating.git

# &nbsp;   cd house\_heating

# &nbsp;   ```

# 

# 2\.  \*\*Create and activate a virtual environment (recommended):\*\*

# &nbsp;   ```bash

# &nbsp;   python -m venv venv

# &nbsp;   source venv/bin/activate  # On Windows, use: `venv\\Scripts\\activate`

# &nbsp;   ```

# 

# 3\.  \*\*Install the required dependencies:\*\*

# &nbsp;   ```bash

# &nbsp;   pip install -r requirements.txt

# &nbsp;   ```

# 

# \### Running the Simulation

# 

# The simulation is controlled by a set of parameters defined in a Python dictionary. You can run experiments by creating a script (e.g., `pipeline/experiments.py`) that imports the `HeatingModel`, defines the parameters, and runs the simulation.

# 

# \#### 1. Configure Model Parameters

# 

# The model's behavior is defined by a single dictionary. This dictionary describes the house layout, physical properties, and initial conditions.

# 

# \*Example `model\_parameters` structure:\*```python

# model\_parameters = {

# &nbsp;   "areas": {

# &nbsp;       "A1": {

# &nbsp;           "row\_min": 0, "row\_max": 33, "col\_min": 0, "col\_max": 50,

# &nbsp;           "init\_func": lambda x: 285 + 5 \* np.random.random(x.shape),

# &nbsp;           "desired\_temp": 300

# &nbsp;       },

# &nbsp;       # ... other area definitions ...

# &nbsp;   },

# &nbsp;   "walls": {

# &nbsp;       # ... wall definitions ...

# &nbsp;   },

# &nbsp;   "doors": {

# &nbsp;       # ... door definitions ...

# &nbsp;   },

# &nbsp;   "windows": {

# &nbsp;       # ... window definitions ...

# &nbsp;   },

# &nbsp;   "radiators": {

# &nbsp;       # ... radiator definitions ...

# &nbsp;   },

# &nbsp;   "domain": {

# &nbsp;       "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))\[0], "dx": 1

# &nbsp;   },

# &nbsp;   "force\_term": lambda x, t, mask: np.where(mask == 1, 1.5, ...),

# &nbsp;   "window\_temp": lambda t: 270 - 5 \* np.sin(24 \* t / 3600),

# &nbsp;   "diffusion\_coefficient": 0.5,

# &nbsp;   "current\_time": 0.0

# }```

# 

# \*\*Parameter Descriptions:\*\*

# \*   `areas`: Defines the rooms. Each area requires:

# &nbsp;   \*   `row\_min`, `row\_max`, `col\_min`, `col\_max`: The grid boundaries of the area.

# &nbsp;   \*   `init\_func`: A lambda function to set the initial temperature.

# &nbsp;   \*   `desired\_temp`: The target temperature (in Kelvin) for the room's thermostat.

# \*   `walls`, `doors`, `windows`, `radiators`: Dictionaries defining the grid coordinates for each of these elements.

# \*   `domain`: Contains the grid information. `dx` is the spatial step size.

# \*   `force\_term`: A lambda function that defines the heat output of the radiators based on the radiator mask.

# \*   `window\_temp`: A lambda function that defines the external temperature at the windows over time `t`.

# \*   `diffusion\_coefficient`: The thermal diffusivity constant (`α`) for the heat equation.

# \*   `current\_time`: The starting time for the simulation.

# 

# \#### 2. Run an Experiment Script

# 

# Create a Python script (e.g., `pipeline/experiments.py`) to run the simulation.

# 

# \*Example `experiments.py`:\*

# ```python

# import numpy as np

# from pipeline.main import HeatingModel

# 

# \# Define the model\_parameters dictionary as shown above

# \# Or, load it from a JSON file:

# \# model = HeatingModel({}).load\_params\_from\_file("data/model\_params.json")

# 

# model\_parameters = {

# &nbsp;   # ... Paste the full parameter dictionary here ...

# }

# 

# \# 1. Initialize the model

# model = HeatingModel(model\_parameters)

# 

# \# 2. Evolve the simulation for a number of steps

# \# E.g., 5000 steps with a time-step of 1.0

# model.evolve(n\_steps=5000, dt=1.0)

# 

# \# 3. Generate and display an image of the final state

# final\_frame = model.build\_image\_frame()

# final\_frame.show()

# 

# \# To save the frame:

# \# final\_frame.save("results/final\_temperature\_distribution.png")```

# 

# To run the script, execute the following command in your terminal:

# ```bash

# python pipeline/experiments.py```

# The script will initialize the model, run the simulation, and then display an image representing the final temperature distribution.

# 

# \### Visualizing the House Layout

# 

# The `build\_image\_frame()` method generates an image of the house, where colors represent different elements:

# 

# \*   \*\*Walls:\*\* Black

# \*   \*\*Windows:\*\* Blue

# \*   \*\*Doors:\*\* Brown

# \*   \*\*Rooms (Areas):\*\* Colored based on temperature using a cool-to-warm colormap.

# 

# \*(Here you can add an image of the house layout generated by the model for clarity)\*

