"""
Classes and functions for clustering artworks.
"""

import clip
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, DBSCAN

import numpy as np
import pandas as pd
import pickle
import os


device = "cuda" if torch.cuda.is_available() else "cpu"



class EmbeddingDatasetBuilder:

    pass



class ArtworkClusterer:
    
    pass

