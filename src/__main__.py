"""
Run the experiments from here.
"""

from topicmodel import TopicModel
from describewithLLM import Descriptor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle



def main():
    # 0. experiments setup
    nr_topics_range = range(3, 29)
    results = {"kmeans": {True: {}, False: {}}, "fcmeans": {True: {}, False: {}}}
    metrics = ["TD", "IEPS", "IEC", "CES", "Inertia"]
    for m in results.keys():
        for r in results[m].keys():
            for metric in metrics:
                results[m][r][metric] = []
    
    # 1. running the experiments
    descriptor = Descriptor()
    for nr_topics in nr_topics_range:
        tm = TopicModel(nr_topics=nr_topics)
        for m in results.keys():
            for r in results[m].keys():
                _, image_topics, scores = tm.fit(method=m, reduce=r)
                _, scores["CES"] = descriptor(tm.output_dir, image_topics)
                if "Inertia" not in scores:
                    scores["Inertia"] = 0
                for metric in metrics:
                    results[m][r][metric].append(scores[metric])

    # 2. plotting and saving the results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    plot_results(results, nr_topics_range, metrics)

def plot_results(results, nr_topics_range, metrics):
    """Plots the results.

    Args:
        results (dict): The results dictionary.
        nr_topics_range (range): The number of topics range.
        metrics (list): The metrics to plot.
    """
    sns.set_style("whitegrid")
    colors = ["crimson", "darkorange", "dodgerblue", "seagreen"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        color_idx = 0
        for m in results.keys():
            for r in results[m].keys():
                mean_value = np.mean(results[m][r][metric])
                std_value = np.std(results[m][r][metric])
                label = f"{m.capitalize()} (μ={mean_value:.2f}, std={std_value:.2f})"
                label = f"UMAP+{label}" if r else label
                plt.plot(nr_topics_range, results[m][r][metric], label=label, color=colors[color_idx], linewidth=1.7)
                color_idx += 1
        
        plt.xlabel("Number of Topics")
        plt.ylabel(metric)
        plt.legend()
        plt.title(f"Trend of {metric} over Number of Topics")
        plt.savefig(f"{metric}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    main()
