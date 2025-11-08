import os
import pickle
import numpy as np

def get_file_names(directory):
    """
    directory: path to the directory containing csv files
    Returns a list of file names in the directory that end with '.csv'.
    """
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_names.append(os.path.join(root, file))
    return file_names

def read_csv_file(M, file_name):
    """
        M: nominal mass of the quadrotor
        file_name: path to the csv file

        Extracts the states and control inputs from the csv file,
        calculates the uncertainty \mathcal{H} (Eq. 3 in the paper),
        and returns the states, actions (uncertainty), and terminal flags.
    """
    try:
        # Use genfromtxt for robust CSV loading, skipping the header row.
        # Assumes your columns A, B, C... start at index 0, 1, 2...
        data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None, None, None

    try:
        # --- Data Mapping (based on your new 22-column format) ---
        
        # Columns B, C (indices 1, 2)
        secs = data[:, 1]
        nsecs = data[:, 2] * 1e-9
        t = secs + nsecs
        t = t - t[0]  # Offset by the first entry

        # Columns D, E, F (indices 3, 4, 5)
        x_curr = data[:, 3]
        y_curr = data[:, 4]
        z_curr = data[:, 5]

        # Columns G, H, I (indices 6, 7, 8)
        x_dot_curr = data[:, 6]
        y_dot_curr = data[:, 7]
        z_dot_curr = data[:, 8]

        # Columns J, K, L (indices 9, 10, 11)
        x_ddot_curr = data[:, 9]
        y_ddot_curr = data[:, 10]
        z_ddot_curr = data[:, 11]

        # Columns M, N, O (indices 12, 13, 14) - NO CONVERSION NEEDED!
        roll_curr = data[:, 12]
        pitch_curr = data[:, 13]
        yaw_curr = data[:, 14]

        # Columns P, Q, R (indices 15, 16, 17)
        # Note: These are available but not used in the original script's state/action calculation
        # roll_dot_curr = data[:, 15]
        # pitch_dot_curr = data[:, 16]
        # yaw_dot_curr = data[:, 17]

        # Columns S, T, U (indices 18, 19, 20) - NO CALCULATION NEEDED!
        tau_x = data[:, 18]
        tau_y = data[:, 19]
        tau_z = data[:, 20]
        
        # Column V (index 21)
        # collective_thrust = data[:, 21]

        # Create previous-timestep versions of torques for the state vector
        # Using np.roll is a clean way to shift the array
        tau_x_prev = np.roll(tau_x, 1)
        tau_x_prev[0] = 0  # Set the first element to 0 as there's no previous step
        tau_y_prev = np.roll(tau_y, 1)
        tau_y_prev[0] = 0
        tau_z_prev = np.roll(tau_z, 1)
        tau_z_prev[0] = 0

    except IndexError as e:
        raise IndexError(f"CSV file '{file_name}' seems to have fewer columns than expected: {e}")
    except ValueError as e:
        raise ValueError(f"Data in '{file_name}' cannot be converted to float. Check for non-numeric values: {e}")

    terminal = np.zeros(len(t), dtype='bool')
    terminal[-1] = 1

    # State vector, faithfully replicating the structure from the original script
    states = np.array([x_curr, y_curr, z_curr,
                        x_dot_curr, y_dot_curr, z_dot_curr,
                        x_ddot_curr, y_ddot_curr, z_ddot_curr,
                        roll_curr, pitch_curr, yaw_curr,
                        tau_x_prev, tau_y_prev, tau_z_prev,])

    # Action (residual H), faithfully replicating the original script's formula
    # Note: As we discussed, this formula mixes rotational inputs (torques) with a linear model.
    # We are preserving it to match what the training script expects.
    p_ddot = np.array([x_ddot_curr, y_ddot_curr, z_ddot_curr])
    tau = np.array([tau_x, tau_y, tau_z])
    actions = tau - M * np.identity(tau.shape[0]) @  p_ddot

    return states, actions, terminal

def prepare_data():
    dataset_dir = 'dataset_drone' # IMPORTANT: Make sure this points to your folder of new CSVs
    file_names = get_file_names(dataset_dir)
    if not file_names:
        print(f"Error: No .csv files found in the '{dataset_dir}' directory.")
        return
        
    data = {'states': [], 'actions': [], 'terminal': []}
    max_length = 0
    # The nominal mass M used in the calculation. Change if your drone's mass is different.
    nominal_mass = 1.0 

    for file_name in file_names:
        print(f"Processing {file_name}...")
        states, actions, terminals = read_csv_file(nominal_mass, file_name)
        if states is None:
            continue
        # Transpose states and actions to be (num_points, num_features)
        data['states'].append(states.T)
        data['actions'].append(actions.T)
        data['terminal'].append(terminals)
        max_length = max(max_length, states.shape[-1])

    if not data['states']:
        print("Error: No data was successfully processed. Halting.")
        return

    # Concatenate all trajectories into single large arrays
    for key in data.keys():
        data[key] = np.concatenate(data[key], axis=0)
    
    output_path = os.path.join(dataset_dir, 'data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("\n--------------------------------------------------")
    print(f"Data preparation complete.")
    print(f"Saved processed data to {output_path}")
    print(f"Please set `max_path_length` to {max_length} in your config file.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    prepare_data()