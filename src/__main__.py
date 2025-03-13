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
    nr_topics_range = range(3, 21)
    results = {"kmeans": {True: {}, False: {}}, "birch": {True: {}, False: {}}}
    metrics = ["TD", "IEPS", "IEC", "CES"]
    for m in results.keys():
        for r in results[m].keys():
            for metric in metrics:
                results[m][r][metric] = []
    
    # 1. running the experiments
    descriptor = Descriptor()
    for nr_topics in nr_topics_range:
        for m in results.keys():
            for r in results[m].keys():
                print(f"Experiment with {m.upper()} and UMAP={r}.")
                tm = TopicModel(nr_topics=nr_topics)
                _, image_topics, scores = tm.fit(method=m, reduce=r)
                _, scores["CES"] = descriptor(tm.output_dir, image_topics)
                for metric in metrics:
                    results[m][r][metric].append(scores[metric])

    # 2. plotting and saving the results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    plot_results(results, nr_topics_range, metrics)
    plot_summary_table(results, metrics)


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
                label = f"UMAP+{m.upper()}" if r else f"{m.upper()}"
                plt.plot(nr_topics_range, results[m][r][metric], label=label, color=colors[color_idx], linewidth=1.5)
                color_idx += 1
        
        plt.xlabel("Number of Topics")
        plt.ylabel(metric)
        plt.legend()
        plt.title(f"Trend of {metric} over Number of Topics")
        plt.savefig(f"{metric}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_summary_table(results, metrics):
    """Plots the summary table.

    Args:
        results (dict): The results dictionary.
        metrics (list): The metrics to plot.
    """
    summary_data = []
    for m in results.keys():
        for r in results[m].keys():
            name = f"UMAP+{m.upper()}" if r else f"{m.upper()}"
            row = [name]
            for metric in metrics:
                mean_value = np.mean(results[m][r][metric])
                std_value = np.std(results[m][r][metric])
                row.append(f"{mean_value:.2f} ± {std_value:.2f}")
            summary_data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=summary_data, colLabels=["Method"] + metrics, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.auto_set_column_width([i for i in range(len(metrics) + 1)])
    plt.title("Summary of Metrics (Mean ± Std Dev)")
    plt.savefig("summary_table.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()
