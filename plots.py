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
            perf = df[df["Defense"]=="Channel Shuffling"]["Performance"]
            df["Model"]=file.split("_")[1]
            df["Plen"]=file.split("_")[2][:-4]
            df["Accuracy Ratio"] = df["Performance"]/perf.values[0]
            df["Accuracy Ratio"] = df["Accuracy Ratio"].apply(lambda x: min(x, 1))
            dfs.append(df)
    merged_df = pd.concat(dfs)
    merged_df.drop(columns=["Unnamed: 0"], inplace=True)
    return merged_df

def table():
    data = aggregate_data()
    print(data.groupby(["Defense", "Model"])[["Integrity", "Accuracy Ratio"]].mean())

def plot():
    data = aggregate_data()
    g = sns.FacetGrid(data, col="Plen", row="Model", margin_titles=True)
    g.map_dataframe(sns.scatterplot, x="Integrity", y="Accuracy Ratio", hue="Defense")
    g.add_legend()
    plt.show()

if __name__ == '__main__':
    table()