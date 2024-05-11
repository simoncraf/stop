# Stochastic Team Orienteering Problem (STOP)
Recreation of the simheuristic algorithm for the Stochastic Team Orienteering Problem (STOP), combining a Variable Neighborhood Search (VNS) metaheuristic with Monte Carlo simulation to maximize surveillance drone rewards under uncertain travel times.

## Problem Description
The Stochastic Team Orienteering Problem (STOP) is a variant of the Orienteering Problem (OP) where a team of drones is used to maximize the rewards collected from a set of targets. The drones have a limited flight range and can only visit a subset of targets. The travel times between targets are uncertain and are modeled as random variables. The goal is to find the subset of targets to visit and the order in which to visit them to maximize the expected rewards collected by the drones.

## Source
This code is a recreation of the algorithm proposed in the paper:

```bibtex
@article{panadero2020maximising,
  title={Maximising reward from a team of surveillance drones: A simheuristic approach to the stochastic team orienteering problem},
  author={Panadero, Javier and Juan, Angel A and Bayliss, Christopher and Currie, Christine},
  journal={European Journal of Industrial Engineering},
  volume={14},
  number={4},
  pages={485--516},
  year={2020},
  publisher={Inderscience Publishers (IEL)}
}
```

## Dataset

The dataset used is the same as the one used in the original paper.

### Format

The dataset is structured in the following format:

- **First Three Lines**:
  - `n N`: Total number of vertices.
  - `m P`: Number of paths.
  - `tmax Tmax`: Available time budget per path.

  **Where:**
  - `N`: Number of vertices.
  - `P`: Number of paths.
  - `Tmax`: Time budget available per path.

- **Remaining Lines**:
  Contain data points with the following structure for each point:

  - `x y S`

  **Where:**
  - `x`: X coordinate.
  - `y`: Y coordinate.
  - `S`: Score.

### Remarks

- The first point is the starting point.
- The last point is the ending point.
- The Euclidean distance is used.