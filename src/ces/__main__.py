import pandas as pd

from .utils import clean_data, multinomial_test, umap_embed, visualize_umap


def main() -> None:
    # load data
    dfs = []
    for filename in [
        "data/cluster_result_by_dominantbp_spycas9.csv",
        "data/cluster_result_by_dominantbp_spymac.csv",
        "data/cluster_result_by_dominantbp_ispymac.csv",
    ]:
        dfs.append(clean_data(filename))
    df = pd.concat(dfs).reset_index(drop=True)

    # multinomial test
    df["mtest"] = df.apply(multinomial_test, axis=1)

    # umap
    df["umap_x"], df["umap_y"] = umap_embed(df)
    df.to_csv("result/output.csv", index=False)
    visualize_umap(df["umap_x"], df["umap_y"], df["group"], "result/umap_group.pdf")
    visualize_umap(
        df["umap_x"], df["umap_y"], df["dominant_bp"], "result/umap_dominant_bp.pdf"
    )


if __name__ == "__main__":
    main()
