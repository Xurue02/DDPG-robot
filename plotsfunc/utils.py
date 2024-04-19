import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

def load_pickle_file(data):
    with open(f'{data}.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data