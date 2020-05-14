import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def preprocess_trajectory_data(trajectory, dim, order):
    return np.hstack([trajectory[:, i:(order+1)*3:3] for i in range(dim)])


def get_resource_path():
    return r"https://raw.githubusercontent.com/PetteriPulkkinen/TrackingSimPy/master/trackingsimpy/trajectories" \
           r"/resources/"


def get_file_list():
    resp = requests.get(r'https://github.com/PetteriPulkkinen/TrackingSimPy/tree/master/trackingsimpy/trajectories'
                        r'/resources')
    soup = BeautifulSoup(resp.content, features="html.parser")
    files = [item.text.strip() for item in soup.select("tbody td.content span a")]
    return files


def load_trajectory(filename, order, dim=None):
    """Loads target trajectory from GitHub repository.

    :param dim: Dimension to be used from the trajectory file (max. 3D)
    :param filename: The trajectory file to be loaded from the resource folder
    :param order: Desired order for the state variable (max. 2)
    :dim dim: Desired dimension for the trajectory
    """
    resource_url = get_resource_path()
    df = pd.read_csv(resource_url + filename)
    trajectory = df.to_numpy()
    if dim is None:
        dim = trajectory.shape[1]
    return preprocess_trajectory_data(trajectory, dim, order)
