# Friction-GK
Computing friction coefficient using Green-Kubo formula


## About the code

This code uses output files of the summed forces in from LAMMPS MD simulations to compute the effective friction coefficient at a liquid-solid interface. 

The example simulation input files can be found in the `simulations` folder.

For the equations used for the calculation of an example, see the notebook in the `example-calculation` folder.

## Citation

For the theory and application of the Green-Kubo formula, please see:

***A. T. Bui, S. J. Cox, "Flow is slow at the nanoscale: Revisiting the Green-Kubo relation for friction"
(2024)***

Links to: [Preprint](https://doi.org/10.48550/arXiv.2409.07134)

## Installation

Requirements:
- Python
- Numpy
- Scipy
- Pandas
- Argparse
- Matplotlib
- tqdm

Simply clone the repository:
```sh
git clone https://github.com/annatbui/friction-GK
```

