"""
Main file for clustering artworks.
"""

import argparse
from artwork_clustering import EmbeddingDatasetBuilder, ArtworkClusterer


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model", type=str, default="ViT-B/32")
    parser.add_argument("-fm", "--finetuned_model", type=str, default="models/finetuned-v2.pt")
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--n_terms", type=int, default=10)

    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--reduce_with", type=str, default=None)
    parser.add_argument("--n_components", type=int, default=32)
    parser.add_argument("--represent_with", type=str, default="centroid")

    args = parser.parse_args()

    # 1. initialize the builder
    builder = EmbeddingDatasetBuilder(
        base_model=args.base_model,
        model_path=args.finetuned_model,
        use_base_model=args.use_base_model
    )
    finetuned_model = args.finetuned_model if not args.use_base_model else None

    # 2. initialize the clusterer
    clusterer = ArtworkClusterer(
        base_model=args.base_model,
        model_path=args.finetuned_model,
        dataset=builder(),
        signifiers_path="data/signifiers.pkl"
    )

    # 3. perform clustering
    clusterer.cluster(
        method=args.method,
        reduce_with=args.reduce_with,
        represent_with=args.represent_with,
        n_terms=args.n_terms,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
        n_components=args.n_components
    )
