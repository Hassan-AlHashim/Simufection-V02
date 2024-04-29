
# Project Title

This Python project includes various modules that work together to simulate, process, and visualize data dynamically through a web-based dashboard. The idea is to provide a user friendly interface for SIRD simulation. This project also allows for AI assisted interaction with the results of the simuation data. 

## Modules Description

1. **dash_frontend.py**:
   - This script initializes a web-based dashboard using Dash by Plotly. It is responsible for creating the user interface and integrating other modules to display simulation results.

2. **run_pipeline.py**:
   - Handles the execution of the simulation pipeline. This script ties together input processing, running simulations, and preparing outputs for visualization.

3. **sim_class.py**:
   - Contains the `Simulation` class that models the core functionality of the simulation. It includes methods for initializing the simulation, running iterations, and applying various mathematical models and solvers.

4. **supporting_functions.py**:
   - Provides additional utility functions that support data handling and operations across the simulation and visualization processes. Functions include data formatting, mathematical operations, and custom analytics.

## Installation

To set up this project, you'll need Python 3.6 or later. Install the required libraries using:

```bash
pip install -r requirements.txt
```

Ensure you create a `requirements.txt` file listing all necessary libraries, such as `dash`, `numpy`, `pandas`, etc.

## Usage

To run the dashboard:

```bash
python dash_frontend.py
```

To execute the simulation pipeline:

```bash
python run_pipeline.py
```

## License

This project is licensed under the MIT License

## Demo Video 
https://www.youtube.com/watch?v=hxAVFvufePM


