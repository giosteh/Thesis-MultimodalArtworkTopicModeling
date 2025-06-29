"""
Run the experiments from here.
"""

from describewithLLM import Descriptor
from topicmodel import TopicModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pickle


NR_TOPICS_RANGE = range(2, 21, 2)
METRICS = ["TD", "IEPS", "IEC", "DESv1", "DESv2"]


def modeling():
    # 0. experiments setup
    results = {
        "kmeans": {False: {}, True: {}},
        "birch": {True: {}},
        "fcmeans": {True: {}}
    }
    for m in results.keys():
        for r in results[m].keys():
            results[m][r]["dir"] = []
            results[m][r]["topics"] = []
            results[m][r]["images"] = []
            for metric in METRICS:
                results[m][r][metric] = []

    # 1. running the experiments
    for nr_topics in NR_TOPICS_RANGE:
        model = TopicModel(nr_topics=nr_topics)
        for m in results.keys():
            for r in results[m].keys():
                experiment_name = f"UMAP+{m.upper()}" if r else f"{m.upper()}"
                print(f"\n<Running {experiment_name} with {nr_topics} topics>")

                topics, _, images_top, scores = model.fit(method=m, reduce=r)
                results[m][r]["dir"].append(model.output_dir)
                results[m][r]["topics"].append(topics)
                results[m][r]["images"].append(images_top)
                for metric in METRICS[:-2]:
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
            for d, t, i in zip(results[m][r]["dir"], results[m][r]["topics"], results[m][r]["images"]):
                _, score_v1 = descriptor(output_dir=d, image_paths=i)
                _, score_v2 = descriptor(output_dir=d, image_paths=i, topics=t)
                results[m][r]["DESv1"].append(score_v1)
                results[m][r]["DESv2"].append(score_v2)
    
    # 2. plotting the results
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
        plt.title(f"Trend of {metric}")
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
    
    _, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=summary_data, colLabels=["Method"] + metrics, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)
    table.auto_set_column_width([i for i in range(len(metrics) + 1)])
    plt.savefig("output/summary.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--describe", action="store_true")
    args = parser.parse_args()

    if not args.describe:
        # topic modeling
        modeling()
    else:
        # topic describing
        describing()
