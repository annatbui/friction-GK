import argparse
import numpy as np
import math
from scipy import integrate
from tqdm import tqdm
import pandas as pd



class Constants:
    # Conversion factors
    KCAL2JOULE = 4.1839953808691 * 1000  # unit is J/Kcal
    ANGSTROM2METER = 1e-10  # unit is m/A
    FEMTOSECOND2SECOND = 1e-15  # unit is fs/s
    
    # Scientific constants
    BOLTZMANN = 1.38064852e-23  # unit is J/K
    AVOGADRO = 6.0221409e23  # unit is 1/mol


class UnphysicalValue(Exception):
    pass


def get_user_inputs():
    """
    Parses command-line arguments for the compute_friction_GK.py script.

    This function is designed to parse the command-line arguments provided by the user. 
    It sets up the parameters required for computing the friction coefficient.

    Returns:
        argparse.Namespace: An object containing all the command-line arguments. The attributes of this object are:
            - input_file (str): Path to the input file containing the summed forces.
            - data_file (str): Path to the LAMMPS data file containing the box size.
            - output_file (str): Path to the output file where results will be written. Defaults to 'friction_GK.txt'.
            - start_time (int): The start time for computing the friction coefficient. Defaults to 1.
            - end_time (int): The end time for the computation. Defaults to -1, indicating the end of the file.
            - timestep (float): The time in fs between written forces. Defaults to 1.
            - total_correlation_time (int): The length in fs of the correlation. Defaults to 10000.
            - block_number (int): The number of blocks for block averaging. Defaults to 100.
            - temperature (float): The temperature in Kelvin for the computation. Defaults to 300.

    Note:
        This function requires the argparse module. Ensure it is imported before calling this function.
    """
    parser = argparse.ArgumentParser(
        description="Computing the friction coefficient from the Green-Kubo relation."
    )

    parser.add_argument("-ifile", "--input_file",
        required=True, type=str, default=None,
        help="the relative path to the file containing the summed forces.",
    )

    parser.add_argument("-ofile", "--output_file",
        required=False, type=str, default="friction_GK.txt",
        help="name of the output file with the results",
    )

    parser.add_argument(
        "-data", "--data_file",
        required=True, type=str, default=None,
        help="the relative path to the data file containing the box size",
    )

    parser.add_argument(
        "-temp", "--temperature",
        required=False, type=float, default=300,
        help="temperature",
    )

    parser.add_argument(
        "-dt", "--timestep",
        required=False, type=float, default=1,
        help="time in fs between written summed forces",
    )

    parser.add_argument(
        "-t0", "--start_time", 
        required=False, type=str, default=1,
        help="start time to compute the friction coefficient.",
    )

    parser.add_argument(
        "-tend", "--end_time",
        required=False, type=str, default=-1,
        help="end time to compute the friction coefficient.",
    )

    parser.add_argument(
        "-corrlength", "--total_correlation_time",
        required=False, type=float, default=10000,
        help="length in fs of correlation.",
    )

    parser.add_argument(
        "-nblock", "--block_number",
        required=False, type=int, default=100,
        help="number of blocks for block averaging",
    )

    return parser.parse_args()


def get_frames(start_time, end_time, dt_frame, total_frames):
    """
    This function calculates the start and end frames for analysis based on the provided start and end times,
    the time between frames, and the maximum length of the data file. It ensures that the end frame does not
    exceed the maximum length of the data file.

    Parameters:
        start_time (int): Start time for analysis in fs.
        end_time (int): End time for analysis in fs. A value of -1 indicates the end of the data.
        dt_frame (float): Time in fs between consecutive frames.
        total_frames (int): Total number of frames in the data file.

    Returns:
        tuple: A tuple containing the start and end frames for analysis (both inclusive).
    """
    # Calculate start frame based on start time and time between frames
    start_frame = max(0, int(int(start_time) / dt_frame))
    
    # Calculate end frame based on end time, time between frames, and max length of the data file
    if end_time == -1:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, int(int(end_time) / dt_frame))
    
    return start_frame, end_frame


def compute_prefactor(path_to_datafile: str, temperature: float):
    """
    Compute prefactor for friction coefficient. Makes sure the units are correct in the end.
    
    Arguments:
        path_to_datafile (str) : Path to data file.
        temperature (float) : Temperature in K.
    
    Returns:
        prefactor (float): beta/area for friction.
    """
    x, y = None, None  # Initialize x and y to ensure they are defined

    try:
        with open(path_to_datafile, 'r') as f:
            for line in f:
                if 'xlo xhi' in line:
                    x = float(line.split()[1])  # Extract x dimension
                elif 'ylo yhi' in line:
                    y = float(line.split()[1])  # Extract y dimension
                if x is not None and y is not None:
                    break  # Exit loop if both dimensions are found
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file {path_to_datafile} not found.")

    if x is None or y is None:
        raise ValueError("File does not contain necessary dimensions ('xlo xhi' or 'ylo yhi').")

    surface_area_solid = x * y  # Calculate surface area

    # Compute and return the prefactor
    return (
        (Constants.KCAL2JOULE / Constants.ANGSTROM2METER / Constants.AVOGADRO) ** 2
        / 2 # average over x and y direction
        * Constants.FEMTOSECOND2SECOND
        / Constants.BOLTZMANN
        / temperature
        / surface_area_solid
        / Constants.ANGSTROM2METER ** 2
    )



def compute_friction_GK(
    path_to_forces: str,
    correlation_time: float,
    block_number: int,
    start_time: int,
    end_time: int,
    dt_frame: float,
    prefactor: float,
):

    """
    Compute mean friction coefficient lambda via Green-Kubo relation from summed force autocorrelation.

    Arguments:
        path_to_forces (str) : Path to file where forces are stored.
        correlation_time (float) : Time (in fs) to correlate the forces, usually 1000 fs is sufficient.
        block_number (int) : Number of blocks used for statistical error analysis.
        start_time (int) : Start time for analysis (in frames).
        end_time (int) : End time for analysis (in frames).
        dt_frame (float): Time (in fs) between frames where summed force was measured.
        prefactor (float) : Prefactor beta/area.

    Returns:
        np.ndarray: Array containing time, average autocorrelation, standard error of the autocorrelation, 
        average friction coefficient, and standard error of friction coefficient.
    """


    forces = pd.read_csv(
        path_to_forces, skiprows=1, header=None, delim_whitespace=True, usecols=[0,1]
    ).to_numpy()
    


    # Determine frames to samples over
    start_frame, end_frame = get_frames(
        start_time, end_time, dt_frame, len(forces)
    )

    # Calculate correlation frames
    num_correlation_frames = int(correlation_time / dt_frame)
    num_samples = int((end_frame - start_frame))


    if num_correlation_frames >= num_samples:
        raise UnphysicalValue(
            f"Correlation frames ({num_correlation_frames}) exceed the number of available samples ({num_samples})."
        )


    # Allocate arrays for calculations
    acf_total = np.zeros(num_correlation_frames)
    num_samples_correlated = np.zeros(num_correlation_frames)
    acf_blocks = np.zeros((block_number, num_correlation_frames))

    samples_per_block = math.ceil(num_samples / block_number)
    


    if samples_per_block < num_correlation_frames:
        raise UnphysicalValue(
            f"Samples per block ({samples_per_block}) are less than correlation frames ({num_correlation_frames})."
        )

    # Determine part to loop about
    force_data = forces[start_frame:end_frame]
    index_current_block_used = 0
    previous_num_samples_correlated = np.zeros(num_correlation_frames)
    
    # Main loop to compute autocorrelation
    for frame, force in enumerate(tqdm(force_data)):

        # compute last frame sampled, i.e. usually frame+correlation frames
        last_correlation_frame = frame + num_correlation_frames
        if last_correlation_frame > num_samples - 1:
            last_correlation_frame = num_samples

        # save how many frames were used for correlation
        num_frames_correlated = last_correlation_frame - frame
        num_samples_correlated[:num_frames_correlated] += 1

        # compute autocorrelation of summed force per frame, for now for each direction separately
        acf_per_frame = np.sum(
            force_data[frame] * force_data[frame:last_correlation_frame],
            axis=1,
        )

        
        acf_total[:num_frames_correlated] += acf_per_frame
        
        
        # compute block averages
        acf_blocks[index_current_block_used, :num_frames_correlated] += acf_per_frame

        # close block when number of samples per block are reached
        if (
            frame + 1 >= (index_current_block_used + 1) * samples_per_block
            or frame + 1 == num_samples
        ):
            # initialise with 0
            num_samples_correlated_per_block = 0
            # check how many samples per frame were taken for this block
            if index_current_block_used == 0:
                # in first block this corresponds to the global number of samples correlated
                num_samples_correlated_per_block = num_samples_correlated
            else:
                # in all others we just need to get the difference between current and previous global samples
                num_samples_correlated_per_block = num_samples_correlated - previous_num_samples_correlated

                    
            # average current block
            acf_blocks[index_current_block_used, :] = acf_blocks[index_current_block_used, :]/ num_samples_correlated_per_block
            
            # define previous global number of samples
            previous_num_samples_correlated = num_samples_correlated.copy()
            
            # increment index to move to next block
            index_current_block_used += 1
     

    # get average autocorrelation
    avg_acf = acf_total / num_samples_correlated
    std_acf = np.std(acf_blocks, axis=0)


    # get average of friction
    friction_coeff = prefactor * integrate.cumtrapz(
        avg_acf, dx=dt_frame, initial=0.0
    )
    
    friction_coeff_block = prefactor * integrate.cumtrapz(
        acf_blocks, dx=dt_frame, axis=1, initial=0.0
    )

    std_friction_coeff = np.std(friction_coeff_block, axis=0)

    
    
    # arange results in 4 columns
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
    - results_array: A list of lists or a 2D array containing the results data.
    - path_to_output: String specifying the path to the output file.
    """

    df = pd.DataFrame(results_array)

    df.columns = ["Time [fs]",
                         "F(t)*F(0) [(kcal/mole/A)^2]",
                         "error F(t)*F(0) [(kcal/mole/A)^2]",
                         "lambda [N*s/m^3]",
                         "error lambda [N*s/m^3]"]
    
    df.to_csv(path_to_output, index=False, sep='\t', header=True)

if __name__ == "__main__":
    
    # Get arguments from user input
    args = get_user_inputs()
    
    # Define paths and parameters from arguments
    forces_path = args.input_file
    datafile_path = args.data_file
    output_path = args.output_file
    start_time = args.start_time
    end_time = args.end_time
    timestep = args.timestep
    total_correlation_time = args.total_correlation_time
    num_blocks = args.block_number
    temperature = args.temperature

    # Compute the prefactor for the friction coefficient (also accounts for units)
    prefactor = compute_prefactor(datafile_path, temperature)

    # Compute friction based on the autocorrelation function
    results_array = compute_friction_GK(
        forces_path,
        total_correlation_time,
        num_blocks,
        start_time,
        end_time,
        timestep,
        prefactor
    )

    # Write results to the output file
    write_output_to_file(results_array, output_path)