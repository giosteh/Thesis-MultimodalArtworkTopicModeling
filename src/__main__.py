"""
Run the experiments from here.
"""

from topicmodel import TopicModel
from describewithLLM import Descriptor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pickle


NR_TOPICS_RANGE = range(3, 21)
METRICS = ["TD", "IEPS", "IEC", "CES"]


def modeling():
    # 0. experiments setup
    results = {
        "kmeans": {False: {}, True: {}},
        "birch": {False: {}, True: {}},
        "fcmeans": {True: {}}
    }
    for m in results.keys():
        for r in results[m].keys():
            results[m][r]["topics"] = []
            for metric in METRICS[:-1]:
                results[m][r][metric] = []

    # 1. running the experiments
    for nr_topics in NR_TOPICS_RANGE:
        model = TopicModel(nr_topics=nr_topics)
        for m in results.keys():
            for r in results[m].keys():
                experiment_name = f"UMAP+{m.upper()}" if r else f"{m.upper()}"
                print(f"\n<Running {experiment_name} with {nr_topics} topics>")

                _, image_topics, scores = model.fit(method=m, reduce=r)
                results[m][r]["dir"] = model.output_dir
                results[m][r]["topics"].append(image_topics)
                for metric in METRICS[:-1]:
                    results[m][r][metric].append(scores[metric])
    
    # 2. saving the results
    with open("output/results.pkl", "wb") as f:
        pickle.dump(results, f)

def describing():
    # 0. loading the results
    with open("output/results.pkl", "rb") as f:
        results = pickle.load(f)

    # 1. running the experiments
    descriptor = Descriptor()
    for m in results.keys():
        for r in results[m].keys():
            for image_topics in results[m][r]["topics"]:
                output_dir = results[m][r]["dir"]
                _, score = descriptor(output_dir, image_topics)
                results[m][r][METRICS[-1]].append(score)

    # 2. saving the results
    with open("output/results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # 3. plotting the results
    plot_results(results, NR_TOPICS_RANGE, METRICS)
    plot_summary_table(results, METRICS)

def plot_results(results, nr_topics_range, metrics):
    """Plots the results.

    Args:
        results (dict): The results dictionary.
        nr_topics_range (range): The number of topics range.
        metrics (list): The metrics to plot.
    """
    sns.set_style("whitegrid")
    colors = ["crimson", "darkorange", "dodgerblue", "seagreen", "darkviolet"]
    
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
        plt.savefig(f"output/{metric}.png", format="png", dpi=300, bbox_inches="tight")
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
    plt.savefig("output/summary_table.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--describe", action="store_true")
    args = parser.parse_args()

    if not args.describe:
        modeling()
    else:
        describing()
