import os
import pandas as pd

def preprocess_trajectory_data(trajectory, dim, order):
    traj_trunc = trajectory[:, :dim]  # get the data up to desired dimension
    states = np.zeros((traj_trunc.shape[0], traj_trunc.shape[1] * (order+1)))  # Add additional dims based on the order
    states[:, ::(order+1)] = traj_trunc
    return states

def get_resource_path():
    cdir = os.path.dirname(__file__)
    return os.path.join(cdir, 'resources')

def get_file_list():
    files = os.listdir(get_resource_path())
    return files


def load_trajectory(filename, order, dim=None):
    """Loads target trajectory from a given file. Works only for position trajectories.

    :param filename: The trajectory file to be loaded from the resource folder.
    :param order: Desired order for the state variable.
    :dim dim: Desired dimension for the trajectory.
    """
    rdir = get_resource_path()
    df = pd.read_csv(os.path.join(rdir, filename))
    trajectory = df.to_numpy()
    if dim is None:
        dim = trajectory.shape[1]
    return preprocess_trajectory_data(trajectory, dim, order)



if __name__ == '__main__':
    print(__file__)
