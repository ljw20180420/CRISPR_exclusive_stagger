#!/usr/bin/env python

import os

import matplotlib.pyplot as plt

from src.ces import (
    collect_data,
    collect_data2,
    draw_box_bar,
    filter_data,
    multinomial_test,
    pairwise_test,
    pairwise_transform,
    umap_embed,
    visualize_umap,
)


def main() -> None:
    # Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
    plt.switch_backend("agg")

    df = collect_data2("tem_nofilter")

    df = filter_data(
        df,
        sum_x_thres=100,
        mut_fre_x_thres={"spycas9": 0.08, "spymac": 0.02, "ispymac": 0.02},
    )

    for target, yup in zip(
        ["tem_1", "tem_2", "tem_3", "tem_4"], [0.7, 0.15, 0.15, 0.2]
    ):
        pt_df = pairwise_transform(
            df,
            target,
        )

        pairwise_test(
            df=pt_df,
            target=target,
            methods=["t-test-ind", "Welch-test", "Yuen-test", "t-test-rel"],
        )

        draw_box_bar(
            df=pt_df,
            target=target,
            yup=yup,
        )

    # multinomial test
    df["mtest"] = df.apply(multinomial_test, axis=1)

    # umap
    df["umap_x"], df["umap_y"] = umap_embed(df)

    # save
    os.makedirs("result", exist_ok=True)
    df.to_csv("result/output.csv", index=False)

    # visualize umap
    visualize_umap(
        df["umap_x"],
        df["umap_y"],
        df["cas"],
        "result/umap_group.pdf",
    )
    visualize_umap(
        df["umap_x"],
        df["umap_y"],
        df["dominant_bp"],
        "result/umap_dominant_bp.pdf",
    )


if __name__ == "__main__":
    main()
