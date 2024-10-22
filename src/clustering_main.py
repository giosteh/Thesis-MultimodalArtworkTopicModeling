"""
Main file for clustering artworks.
"""

import argparse
from artwork_clustering import EmbeddingDatasetBuilder, ArtworkClusterer


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    