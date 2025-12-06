import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import chisquare
from sklearn.preprocessing import StandardScaler


def clean_data(filename: os.PathLike) -> pd.DataFrame:
    df = (
        pd.read_csv(filename)
        .rename(columns={"5bp": "0bp"})
        .assign(
            dominant_bp=lambda df: df["dominant_bp"].map(
                lambda x: x if x != "5bp" else "0bp"
            )
        )
    )
    columns = df.columns.to_list()
    columns.remove("0bp")
    columns.insert(columns.index("1bp"), "0bp")

    return df.reindex(columns=columns)


def multinomial_test(row: pd.Series) -> float:
    row = row[[f"{bp}bp" for bp in range(5)]]
    chi2_statistic, p_value = chisquare(
        f_obs=row.to_list(), f_exp=[sum(row) / len(row)] * len(row)
    )
    return p_value


def umap_embed(df: pd.DataFrame) -> tuple[np.ndarray]:
    reducer = umap.UMAP()
    data = df[[f"{bp}bp" for bp in range(5)]].to_numpy()
    scaled_data = StandardScaler().fit_transform(data)
    embeddings = reducer.fit_transform(scaled_data)
    return embeddings[:, 0], embeddings[:, 1]


def visualize_umap(
    umap_x: np.ndarray, umap_y: np.ndarray, annots: pd.Series, filename: os.PathLike
) -> None:
    ax = sns.scatterplot(
        x=umap_x,
        y=umap_y,
        hue=annots.astype("category"),
        s=1,
        alpha=0.2,
        edgecolor=None,
    )
    ax.get_figure().savefig(filename)
    plt.close("all")
