import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os


def preprocess_trajectory_data(trajectory, dim, order):
    return np.hstack([trajectory[:, i:(order+1)*3:3] for i in range(dim)])


def get_resource_path(local=True):
    if local:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
    else:
        return r"https://raw.githubusercontent.com/PetteriPulkkinen/TrackingSimPy/master/trackingsimpy/trajectories" \
           r"/resources/"


def get_file_list(local=True):
    if local:
        path = get_resource_path(local=True)
        return sorted(os.listdir(path))
    else:
        resp = requests.get(r'https://github.com/PetteriPulkkinen/TrackingSimPy/tree/master/trackingsimpy/trajectories'
                            r'/resources')
        soup = BeautifulSoup(resp.content, features="html.parser")
        files = [item.text.strip() for item in soup.select("tbody td.content span a")]
        return files


def load_trajectory(filename, order=2, dim=3, local=True):
    """Loads target trajectory from local files or GitHub repository.

    :param local: Whether to load from local files or from GitHub (default True -> local)
    :param dim: Dimension to be used from the trajectory file (max. 3D)
    :param filename: The trajectory file to be loaded from the resource folder
    :param order: Desired order for the state variable (max. 2)
    :dim dim: Desired dimension for the trajectory
    """
    resource_path = get_resource_path(local)
    if local:
        df = pd.read_csv(os.path.join(resource_path, filename))
    else:
        df = pd.read_csv(resource_path + filename)
    trajectory = df.to_numpy()
    if dim is None:
        dim = trajectory.shape[1]
    return preprocess_trajectory_data(trajectory, dim, order)
