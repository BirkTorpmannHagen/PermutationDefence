import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def aggregate_data():
    dfs = []
    for file in os.listdir("."):
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            df["Dataset"]=file.split("_")[0]
            df["Model"]=file.split("_")[1]
            df["Plen"]=file.split("_")[2][:-4]
            dfs.append(df)
    merged_df = pd.concat(dfs)
    return merged_df


def plot():
    data = aggregate_data()
    g = sns.FacetGrid(data, col="Dataset", row="Model", margin_titles=True)
    g.map_dataframe(sns.scatterplot, x="Integrity", y="Performance", hue="Defense")
    plt.show()

if __name__ == '__main__':
    plot()