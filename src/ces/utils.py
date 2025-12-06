import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy.stats import chisquare
from sklearn.preprocessing import StandardScaler


def clean_data(filename: os.PathLike) -> pd.DataFrame:
    # Rename 5bp to 0bp.
    df = (
        pd.read_csv(filename)
        .rename(columns={"5bp": "0bp"})
        .assign(
            dominant_bp=lambda df: df["dominant_bp"].map(
                lambda x: x if x != "5bp" else "0bp"
            )
        )
    )

    # Reorder 0bp to the previous postion of 1bp.
    columns = df.columns.to_list()
    columns.remove("0bp")
    columns.insert(columns.index("1bp"), "0bp")

    # Convert columns group (cas protein) and dominant_bp from object to category.
    df["group"] = df["group"].astype(
        pd.CategoricalDtype(["spycas9", "spymac", "ispymac"], ordered=True)
    )
    df["dominant_bp"] = df["dominant_bp"].astype(
        pd.CategoricalDtype(["0bp", "1bp", "2bp", "3bp", "4bp"], ordered=True)
    )

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
    if not isinstance(annots.dtype, pd.CategoricalDtype):
        annots = annots.astype("category")
    ax = sns.scatterplot(
        x=umap_x,
        y=umap_y,
        hue=annots,
        s=1,
        alpha=0.2,
        edgecolor=None,
    )
    ax.get_figure().savefig(filename)
    plt.close("all")
