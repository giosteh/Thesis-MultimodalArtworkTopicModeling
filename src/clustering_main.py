"""
Main file for clustering artworks.
"""

import argparse
from artwork_clustering import EmbeddingDatasetBuilder, ArtworkClusterer


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model", type=str, default="ViT-B/32")
    parser.add_argument("-fm", "--finetuned_model", type=str, default="models/finetuned.pt")
    parser.add_argument("-md", "--mode", type=str, default="kmeans")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--n_terms", type=int, default=10)
    parser.add_argument("--use_base_model", action="store_true")
    
    args = parser.parse_args()

    # 1. initialize the builder
    builder = EmbeddingDatasetBuilder(
        base_model=args.base_model,
        finetuned_model_path=args.finetuned_model,
        use_base_model=args.use_base_model
    )
    finetuned_model = args.finetuned_model if not args.use_base_model else None

    # 2. initialize the clusterer
    clusterer = ArtworkClusterer(
        base_model=args.base_model,
        finetuned_model_path=finetuned_model,
        dataset=builder()
    )
    
    # 3. cluster the artworks
    clusterer.cluster(
        mode=args.mode,
        n_clusters=args.n_clusters,
        n_terms=args.n_terms
    )
