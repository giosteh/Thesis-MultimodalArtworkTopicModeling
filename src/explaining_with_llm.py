"""
Classes and functions for explaining the clusters using LLMs.
"""

import clip
import torch
import torch.nn as nn

from artwork_clustering import load_model

from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings

# Setting some things up
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"font.family": "Liberation Sans"})


device = "cuda" if torch.cuda.is_available() else "cpu"



