"""
Run the experiments from here.
"""

from topicmodel import TopicModel
from describewithLLM import Descriptor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def main():
    nr_topics_range = range(3, 29)
    # collection of results
    results = {"kmeans": {True: {}, False: {}}, "fcmeans": {True: {}, False: {}}}
    for m in results.keys():
        for r in results[m].keys():
            # initializing each metric list
            results[m][r]["TD"] = []
            results[m][r]["IEPS"] = []
            results[m][r]["IEC"] = []
            results[m][r]["CES"] = []
            results[m][r]["Inertia"] = []
    
    # running the experiments
    descriptor = Descriptor()
    for nr_topics in nr_topics_range:
        tm = TopicModel(nr_topics=nr_topics)
        for m in results.keys():
            for r in results[m].keys():
                _, image_topics, scores = tm.fit(method=m, reduce=r)
                _, scores["CES"] = descriptor(tm.output_dir, image_topics)
                # appending the scores
                for metric in results[m][r].keys():
                    results[m][r][metric].append(scores[metric])
    
    # saving the results
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # plotting the results
    plot_results(results, nr_topics_range)


def plot_results(results, nr_topics_range):
    pass



if __name__ == "__main__":
    main()
