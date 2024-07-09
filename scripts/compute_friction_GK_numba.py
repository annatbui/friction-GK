import argparse
import numpy as np
import math
from scipy import integrate
from tqdm import tqdm
import pandas as pd
from numba import jit, prange

class Constants:
    KCAL2JOULE = 4.1839953808691 * 1000  # unit is J/Kcal
    ANGSTROM2METER = 1e-10  # unit is m/A
    FEMTOSECOND2SECOND = 1e-15  # unit is fs/s
    BOLTZMANN = 1.38064852e-23  # unit is J/K
    AVOGADRO = 6.0221409e23  # unit is 1/mol

class UnphysicalValue(Exception):
    """Exception raised for unphysical values in the computation."""
    pass

def get_user_inputs():
    """
    Parses command-line arguments for the compute_friction_GK.py script.

    Returns:
        argparse.Namespace: An object containing all the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Computing the friction coefficient from the Green-Kubo relation.")
    parser.add_argument("-in", "--input_file", required=True, type=str, help="the relative path to the file containing the summed forces.")
    parser.add_argument("-out", "--output_file", required=False, type=str, default="friction_GK.txt", help="name of the output file with the results")
    parser.add_argument("-data", "--data_file", required=True, type=str, help="the relative path to the data file containing the box size")
    parser.add_argument("-temp", "--temperature", required=False, type=float, default=300, help="temperature")
    parser.add_argument("-dt", "--timestep", required=False, type=float, default=1, help="time in fs between written summed forces")
    parser.add_argument("-t0", "--start_time", required=False, type=int, default=1, help="start time to compute the friction coefficient.")
    parser.add_argument("-tend", "--end_time", required=False, type=int, default=-1, help="end time to compute the friction coefficient.")
    parser.add_argument("-corrlength", "--total_correlation_time", required=False, type=float, default=10000, help="length in fs of correlation.")
    parser.add_argument("-nblock", "--block_number", required=False, type=int, default=100, help="number of blocks for block averaging")
    return parser.parse_args()

def get_frames(start_time, end_time, dt_frame, total_frames):
    """
    Calculates the start and end frames for analysis based on provided times and frame interval.

    Parameters:
        start_time (int): Start time for analysis in fs.
        end_time (int): End time for analysis in fs. -1 indicates the end of the data.
        dt_frame (float): Time interval between frames in fs.
        total_frames (int): Total number of frames in the data.

    Returns:
        tuple: Start and end frames for analysis.
    """
    start_frame = max(0, int(start_time / dt_frame))
    end_frame = total_frames if end_time == -1 else min(total_frames, int(end_time / dt_frame))
    return start_frame, end_frame

def compute_prefactor(path_to_datafile: str, temperature: float):
    """
    Computes the prefactor for the friction coefficient.

    Parameters:
        path_to_datafile (str): Path to the data file containing box size information.
        temperature (float): Temperature in Kelvin.

    Returns:
        float: Computed prefactor for the friction coefficient.

    Raises:
        FileNotFoundError: If the data file is not found.
        ValueError: If the necessary dimensions are not found in the file.
    """
    x, y = None, None  
    try:
        with open(path_to_datafile, 'r') as f:
            for line in f:
                if 'xlo xhi' in line:
                    xlo, xhi = map(float, line.split()[:2])
                    x = xhi - xlo 
                elif 'ylo yhi' in line:
                    ylo, yhi = map(float, line.split()[:2])
                    y = yhi - ylo
                if x is not None and y is not None:
                    break  
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {path_to_datafile} not found.")

    if x is None or y is None:
        raise ValueError("File does not contain necessary dimensions ('xlo xhi' or 'ylo yhi').")

    surface_area_solid = x * y
    return ((Constants.KCAL2JOULE / Constants.ANGSTROM2METER / Constants.AVOGADRO) ** 2
            / 2
            * Constants.FEMTOSECOND2SECOND
            / Constants.BOLTZMANN
            / temperature
            / surface_area_solid
            / Constants.ANGSTROM2METER ** 2)

@jit(nopython=True, parallel=True)
def compute_acf_and_blocks(force_data, num_correlation_frames, block_number, samples_per_block, num_samples):
    """
    Computes the autocorrelation function (ACF) and block averages.

    Parameters:
        force_data (np.ndarray): Array of force data.
        num_correlation_frames (int): Number of correlation frames.
        block_number (int): Number of blocks for averaging.
        samples_per_block (int): Number of samples per block.
        num_samples (int): Total number of samples.

    Returns:
        tuple: Total ACF, number of samples correlated, and block ACFs.
    """
    acf_total = np.zeros(num_correlation_frames)
    num_samples_correlated = np.zeros(num_correlation_frames)
    acf_blocks = np.zeros((block_number, num_correlation_frames))

    for frame in prange(num_samples):
        last_correlation_frame = min(frame + num_correlation_frames, num_samples)
        num_frames_correlated = last_correlation_frame - frame
        num_samples_correlated[:num_frames_correlated] += 1

        acf_per_frame = np.sum(force_data[frame] * force_data[frame:last_correlation_frame], axis=1)
        acf_total[:num_frames_correlated] += acf_per_frame

        block_index = frame // samples_per_block
        acf_blocks[block_index, :num_frames_correlated] += acf_per_frame

    return acf_total, num_samples_correlated, acf_blocks

@jit(nopython=True, parallel=True)
def normalize_acf_blocks(acf_blocks, block_number, samples_per_block, num_correlation_frames, num_samples):
    """
    Normalizes the autocorrelation function (ACF) blocks.

    Parameters:
        acf_blocks (np.ndarray): Block ACFs.
        block_number (int): Number of blocks.
        samples_per_block (int): Number of samples per block.
        num_correlation_frames (int): Number of correlation frames.
        num_samples (int): Total number of samples.

    Returns:
        np.ndarray: Normalized block ACFs.
    """
    for i in prange(block_number):
        start_block = i * samples_per_block
        end_block = min(start_block + samples_per_block, num_samples)
        block_correlated = np.zeros(num_correlation_frames)
        for frame in range(start_block, end_block):
            last_correlation_frame = min(frame + num_correlation_frames, num_samples)
            num_frames_correlated = last_correlation_frame - frame
            block_correlated[:num_frames_correlated] += 1
        acf_blocks[i, :] /= block_correlated
    return acf_blocks

def compute_friction_GK(path_to_forces: str, correlation_time: float, block_number: int, start_time: int, end_time: int, dt_frame: float, prefactor: float):
    """
    Computes the mean friction coefficient using the Green-Kubo relation from summed force autocorrelation.

    Parameters:
        path_to_forces (str): Path to the file containing the summed forces.
        correlation_time (float): Time in fs to correlate the forces.
        block_number (int): Number of blocks for statistical error analysis.
        start_time (int): Start time for analysis in frames.
        end_time (int): End time for analysis in frames.
        dt_frame (float): Time in fs between frames where summed force was measured.
        prefactor (float): Prefactor for the computation.

    Returns:
        np.ndarray: Array containing the time, average autocorrelation, standard error of the autocorrelation,
                    average friction coefficient, and standard error of the friction coefficient.

    Raises:
        UnphysicalValue: If correlation frames exceed the number of available samples.
    """
    forces = pd.read_csv(path_to_forces, skiprows=1, header=None, delim_whitespace=True, usecols=[0, 1]).to_numpy()

    start_frame, end_frame = get_frames(start_time, end_time, dt_frame, len(forces))
    num_correlation_frames = int(correlation_time / dt_frame)
    num_samples = int((end_frame - start_frame))

    if num_correlation_frames >= num_samples:
        raise UnphysicalValue(f"Correlation frames ({num_correlation_frames}) exceed the number of available samples ({num_samples}).")

    samples_per_block = math.ceil(num_samples / block_number)
    if samples_per_block < num_correlation_frames:
        raise UnphysicalValue(f"Samples per block ({samples_per_block}) are less than correlation frames ({num_correlation_frames}).")

    force_data = forces[start_frame:end_frame]

    acf_total, num_samples_correlated, acf_blocks = compute_acf_and_blocks(force_data, num_correlation_frames, block_number, samples_per_block, num_samples)

    avg_acf = acf_total / num_samples_correlated

    acf_blocks = normalize_acf_blocks(acf_blocks, block_number, samples_per_block, num_correlation_frames, num_samples)

    std_acf = np.std(acf_blocks, axis=0)

    friction_coeff_block = np.zeros_like(acf_blocks)
    for i in prange(block_number):
        friction_coeff_block[i, :] = prefactor * integrate.cumtrapz(acf_blocks[i, :], dx=dt_frame, initial=0.0)

    std_friction_coeff = np.std(friction_coeff_block, axis=0)

    friction_coeff = prefactor * integrate.cumtrapz(avg_acf, dx=dt_frame, initial=0.0)

    time = np.arange(len(avg_acf)) * dt_frame
    se_acf = std_acf * 2 / np.sqrt(block_number)
    avg_friction_coeff = friction_coeff
    se_friction_coeff = std_friction_coeff * 2 / np.sqrt(block_number)

    results_array = np.array([time, avg_acf, se_acf, avg_friction_coeff, se_friction_coeff]).T

    return results_array

def write_output_to_file(results_array, path_to_output):
    """
    Writes the computed results to an output file in tab-separated values format.

    Parameters:
        results_array (np.ndarray): Array containing the computed results.
        path_to_output (str): Path to the output file.
    """
    df = pd.DataFrame(results_array)

    time = "t / fs"
    acf = "F(t)*F(0) / (kcal/mole/A)^2"
    se_acf = "err[F(t)*F(0)] / (kcal/mole/A)^2"
    friction = "lambda / N*s/m^3"
    se_friction = "err[lambda] / N*s/m^3"
    
    df.columns = [time, acf, se_acf, friction, se_friction]
    
    df[time] = df[time].map("{:.2f}".format)
    df[acf] = df[acf].map("{:.6f}".format)
    df[se_acf] = df[se_acf].map("{:.6f}".format)
    df[friction] = df[friction].map("{:.6f}".format)
    df[se_friction] = df[se_friction].map("{:.6f}".format)
    
    df.to_csv(path_to_output, index=False, sep='\t', header=True)

if __name__ == "__main__":
    args = get_user_inputs()
    
    forces_path = args.input_file
    datafile_path = args.data_file
    output_path = args.output_file
    start_time = args.start_time
    end_time = args.end_time
    timestep = args.timestep
    total_correlation_time = args.total_correlation_time
    num_blocks = args.block_number
    temperature = args.temperature

    prefactor = compute_prefactor(datafile_path, temperature)

    results_array = compute_friction_GK(forces_path, total_correlation_time, num_blocks, start_time, end_time, timestep, prefactor)

    write_output_to_file(results_array, output_path)
