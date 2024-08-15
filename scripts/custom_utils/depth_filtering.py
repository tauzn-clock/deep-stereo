import numpy as np

def normalise(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())