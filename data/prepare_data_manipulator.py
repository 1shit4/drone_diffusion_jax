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

def read_csv_file(M_drone, file_name):
    """
    M_drone: nominal mass of the quadrotor
    file_name: path to the csv file for the drone-manipulator system

    Extracts states and actions, calculates the residual dynamics H,
    and returns them in the format expected by the training script.
    """
    # ==========================================================================
    # 1. PHYSICAL PARAMETERS OF THE SYSTEM
    # ==========================================================================
    
    # --- Manipulator Parameters (from your measurements) ---
    m1_raw, l1_raw, r1_raw = 45.0, 18.3, 9.15
    m2_raw, l2_raw, r2_raw = 211.0, 18.3, 8.8
    mp = 0.0

    # Convert to SI Units (kilograms and meters)
    m1, l1, r1 = m1_raw / 1000.0, l1_raw / 100.0, r1_raw / 100.0
    m2, l2, r2 = m2_raw / 1000.0, l2_raw / 100.0, r2_raw / 100.0

    # Estimate moments of inertia using the simple rod approximation
    Ic1 = (1/12) * m1 * l1**2
    Ic2 = (1/12) * m2 * l2**2
    
    try:
        data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return None, None, None

    try:
        # ======================================================================
        # 2. DATA MAPPING FROM YOUR CSV FILE (Columns A=0, B=1, etc.)
        # ======================================================================
        
        # Time (Cols B, C)
        t = data[:, 1] + data[:, 2] * 1e-9; t = t - t[0]

        # Drone State (Cols D-O)
        drone_pos = data[:, 3:6]    # [x, y, z]
        drone_vel = data[:, 6:9]    # [vx, vy, vz]
        drone_acc = data[:, 9:12]   # [ax, ay, az]
        drone_rpy = data[:, 12:15]  # [roll, pitch, yaw]
        
        # Manipulator State (Cols Y-AD)
        j1_pos = data[:, 24]  # Col Y
        j2_pos = data[:, 25]  # Col Z (this is theta2)
        j1_vel = data[:, 26]  # Col AA
        j2_vel = data[:, 27]  # Col AB
        j1_acc = data[:, 28]  # Col AC
        j2_acc = data[:, 29]  # Col AD
        
        # Commanded Torques (Cols S-U and W-X)
        drone_tau_cmd = data[:, 18:21] # [tau_x, tau_y, tau_z]
        j1_tau_cmd = data[:, 22]       # Col W
        j2_tau_cmd = data[:, 23]       # Col X

        # ======================================================================
        # 3. CONSTRUCT THE "NAIVE MODEL" AND CALCULATE H
        # ======================================================================
        
        # --- Calculate the state-dependent manipulator mass matrix M_manip ---
        c2 = np.cos(j2_pos) # cos(theta2) for every timestep
        m11 = (Ic1 + m1*r1**2) + Ic2 + m2*(l1**2 + r2**2 + 2*l1*r2*c2) + mp*(l1**2 + l2**2 + 2*l1*l2*c2)
        m12 = Ic2 + m2*(r2**2 + l1*r2*c2) + mp*(l2**2 + l1*l2*c2)
        m22 = (Ic2 + m2*r2**2) + mp*l2**2

        # --- Calculate the "naive" command based on the simple model (`M @ p_ddot`) ---
        
        # Drone part (linear)
        naive_cmd_drone = M_drone * drone_acc
        
        # Manipulator part (rotational, state-dependent)
        p_ddot_manip = np.vstack([j1_acc, j2_acc]).T # Shape: (N, 2)
        naive_cmd_manip = np.zeros_like(p_ddot_manip)
        
        for i in range(len(t)):
            M_manip_i = np.array([[m11[i], m12[i]], [m12[i], m22]])
            naive_cmd_manip[i, :] = M_manip_i @ p_ddot_manip[i, :]
            
        # --- Calculate the residual H (H = tau_real - tau_naive) ---
        
        H_drone = drone_tau_cmd - naive_cmd_drone
        H_manip = np.vstack([j1_tau_cmd, j2_tau_cmd]).T - naive_cmd_manip
        
        # The final 'actions' vector is the full residual H
        actions = np.hstack([H_drone, H_manip])

        # ======================================================================
        # 4. CONSTRUCT THE FINAL STATE AND ACTION VECTORS
        # ======================================================================
        
        # --- Create "previous action" for the state vector ---
        full_tau_cmd = np.hstack([drone_tau_cmd, np.vstack([j1_tau_cmd, j2_tau_cmd]).T])
        prev_full_tau_cmd = np.roll(full_tau_cmd, 1, axis=0)
        prev_full_tau_cmd[0, :] = 0
        
        # --- Assemble the final 21-dimensional state vector ---
        manip_kinematics = np.vstack([j1_pos, j2_pos, j1_vel, j2_vel]).T
        states = np.hstack([
            drone_pos,
            drone_vel,
            drone_acc,
            drone_rpy,
            manip_kinematics,
            prev_full_tau_cmd
        ])
        
        terminal = np.zeros(len(t), dtype='bool')
        terminal[-1] = 1

    except IndexError as e:
        raise IndexError(f"CSV file '{file_name}' has an indexing error. Check that all columns exist. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_name}: {e}")
        return None, None, None
    
    return states, actions, terminal

def prepare_data():
    dataset_dir = 'dataset_new' # IMPORTANT: Make sure this points to your folder of new CSVs
    file_names = get_file_names(dataset_dir)
    if not file_names:
        print(f"Error: No .csv files found in the '{dataset_dir}' directory.")
        return
        
    data = {'states': [], 'actions': [], 'terminal': []}
    max_length = 0
    # The nominal mass M used in the calculation. Change if your drone's mass is different.
    nominal_mass = 0.45

    for file_name in file_names:
        print(f"Processing {file_name}...")
        states, actions, terminals = read_csv_file(nominal_mass, file_name)
        if states is None:
            continue
        data['states'].append(states)
        data['actions'].append(actions)
        data['terminal'].append(terminals)
        max_length = max(max_length, states.shape[0])

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