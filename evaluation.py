import numpy as np
import socket

def evaluate_fn(experiment_folder):
    """A function that calculates fitness values for the individuals that were just infereced
    by the pretrained model. Note: hardcoded for now in the server code.
    :param experiment_folder:

    Args:
        experiment_folder (string): The folder where the inference data is saved in.

    Returns:
        sitting_percentage: float, the percentage of area with comfortable conditions
        dangerous_percentage: float, the percentage of area with dangerous conditions
        sitting: float, the total area with comfortable conditions
        dangerous: float, the total area with comfortable conditions
    """

    # load inference results
    lawson_results = np.load(experiment_folder + '/yearly_lawson.npy')
    total_area = np.load(experiment_folder + '/area_0.npy')

    # calculate comfort category values and comfortable/dangerous conditions
    unique, counts = np.unique(lawson_results, return_counts=True)

    try:
        sitting = np.sum(counts[np.where(unique<=2)[0]])
    except:
        sitting = 0

    try:
        dangerous = np.sum(counts[np.where(unique>=4)[0]])
    except:
        dangerous = 0

    # scale to percentage
    sitting_percentage = (sitting / total_area.item()) * 100
    dangerous_percentage = (dangerous / total_area.item()) * 100

    # scale to 250 x 250 extent (image resolution is 512 x 512, 4.2 times larger)
    sitting = sitting / 4.2
    dangerous = dangerous / 4.2

    return sitting_percentage, dangerous_percentage, sitting, dangerous

def run_inf(port=5559, host='127.0.0.1', timeout=1000):
    """
    Function to run inference on the local server where the pretrained model is running.
    :param port: The port through which the communication with the model happens.
    :param host: The local address of the server.
    :param timeout: A specified amount of time to wait for a response from the server before closing.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host,port))

        s.sendall(b'run model!')

        #send message to server
        data = s.recv(1024).decode()