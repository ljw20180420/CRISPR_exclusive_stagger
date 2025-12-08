import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from plotnine import (
    aes,
    coord_cartesian,
    geom_boxplot,
    geom_col,
    geom_errorbar,
    ggplot,
    scale_color_manual,
    scale_fill_manual,
    scale_x_discrete,
)
from scipy.stats import (
    PermutationMethod,
    binomtest,
    brunnermunzel,
    bws_test,
    chisquare,
    mannwhitneyu,
    ranksums,
    sem,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from sklearn.preprocessing import StandardScaler


def collect_data(data_dir: os.PathLike) -> pd.DataFrame:
    data_dir = pathlib.Path(os.fspath(data_dir))
    dfs = []
    for file in os.listdir(data_dir):
        if (
            not re.search(r"^36t-", file)
            and not re.search(r"^B2-", file)
            and not re.search(r"^A2-", file)
            and not re.search(r"^D2-", file)
            and not re.search(r"^i10t-", file)
        ):
            continue

        if re.search(r"^36t-", file) or re.search(r"^B2-", file):
            cas = "spymac"
        elif re.search(r"^A2-", file) or re.search(r"^D2-", file):
            cas = "spycas9"
        elif re.search(r"^i10t-", file):
            cas = "ispymac"
        else:
            raise ValueError("Unexpected sample")

        mat = re.search(r"([^-]+)-[^-\d](\d)n?+-(\d)-wt(\d)", file)

        dfs.append(
            pd.read_csv(data_dir / file, header=0).assign(
                cas=cas,
                cellline=mat.group(1),
                chip=int(mat.group(2)),
                bio_rep=int(mat.group(3)),
                tech_rep=int(mat.group(4)),
            )
        )

    df = (
        pd.concat(dfs)
        .reset_index(drop=True)
        .assign(mut_x=lambda df: (df["sum_x"] * df["mut_fre_x"]).round())
        .drop(columns=["mut_fre_x"])
    )

    for i in range(1, 11):
        df[f"tem_{i}"] = (df["sum_x"] * df[f"tem_{i}"]).round()
    df["tem_0"] = df["sum_x"] - df[[f"tem_{i}" for i in range(1, 11)]].sum(axis=1)

    # Move tem_0 before tem_1
    columns = df.columns.to_list()
    columns.remove("tem_0")
    columns.insert(columns.index("tem_1"), "tem_0")
    df = df.reindex(columns=columns)

    df = (
        df.groupby(["sgrna", "cas", "cellline", "chip", "bio_rep", "tech_rep"])
        .agg("mean")
        .reset_index(drop=False)
        .groupby(["sgrna", "cas"])
        .agg(
            dict(
                {"sum_x": "sum", "mut_x": "sum"},
                **{f"tem_{i}": "sum" for i in range(0, 11)},
            )
        )
        .reset_index(drop=False)
    ).assign(
        cas=lambda df: df["cas"].astype(
            pd.CategoricalDtype(["spycas9", "spymac", "ispymac"], ordered=True)
        ),
        dominant_bp=df[[f"tem_{bp}" for bp in range(5)]]
        .idxmax(axis=1)
        .astype(pd.CategoricalDtype([f"tem_{bp}" for bp in range(5)], ordered=True)),
        dominant_val=df[[f"tem_{bp}" for bp in range(5)]].max(axis=1),
    )

    return df


def collect_data2(data_dir: os.PathLike) -> pd.DataFrame:
    data_dir = pathlib.Path(os.fspath(data_dir))
    dfs = []
    for file in os.listdir(data_dir):
        if (
            not re.search(r"^36t-", file)
            and not re.search(r"^B2-", file)
            and not re.search(r"^A2-", file)
            and not re.search(r"^D2-", file)
            and not re.search(r"^i10t-", file)
        ):
            continue

        if re.search(r"^36t-", file) or re.search(r"^B2-", file):
            cas = "spymac"
        elif re.search(r"^A2-", file) or re.search(r"^D2-", file):
            cas = "spycas9"
        elif re.search(r"^i10t-", file):
            cas = "ispymac"
        else:
            raise ValueError("Unexpected sample")

        mat = re.search(r"([^-]+)-[^-\d](\d)n?+-(\d)-wt(\d)", file)

        dfs.append(
            pd.read_csv(data_dir / file, header=0).assign(
                cas=cas,
                cellline=mat.group(1),
                chip=int(mat.group(2)),
                bio_rep=int(mat.group(3)),
                tech_rep=int(mat.group(4)),
            )
        )

    df = (
        pd.concat(dfs)
        .reset_index(drop=True)
        .groupby(["sgrna", "cas", "cellline", "chip", "bio_rep", "tech_rep"])
        .agg("mean")
        .reset_index(drop=False)
        .groupby(["sgrna", "cas"])
        .agg(
            dict(
                {"sum_x": "mean", "mut_fre_x": "mean"},
                **{f"tem_{i}": "mean" for i in range(1, 11)},
            )
        )
        .reset_index(drop=False)
    )

    df = df.assign(mut_x=lambda df: (df["sum_x"] * df["mut_fre_x"]).round()).drop(
        columns=["mut_fre_x"]
    )

    for i in range(1, 11):
        df[f"tem_{i}"] = (df["sum_x"] * df[f"tem_{i}"]).round()
    df["tem_0"] = df["sum_x"] - df[[f"tem_{i}" for i in range(1, 11)]].sum(axis=1)

    # Move tem_0 before tem_1
    columns = df.columns.to_list()
    columns.remove("tem_0")
    columns.insert(columns.index("tem_1"), "tem_0")
    df = df.reindex(columns=columns)

    df = df.assign(
        cas=lambda df: df["cas"].astype(
            pd.CategoricalDtype(["spycas9", "spymac", "ispymac"], ordered=True)
        ),
        dominant_bp=df[[f"tem_{bp}" for bp in range(5)]]
        .idxmax(axis=1)
        .astype(pd.CategoricalDtype([f"tem_{bp}" for bp in range(5)], ordered=True)),
        dominant_val=df[[f"tem_{bp}" for bp in range(5)]].max(axis=1),
    )

    return df


def filter_data(
    df: pd.DataFrame, sum_x_thres: float, mut_fre_x_thres: dict
) -> pd.DataFrame:
    return (
        df.query("sum_x > @sum_x_thres")
        .query(
            """
            cas == 'spycas9' and mut_x / sum_x > @mut_fre_x_thres['spycas9'] or \
            cas == 'spymac' and mut_x / sum_x > @mut_fre_x_thres['spymac'] or \
            cas == 'ispymac' and mut_x / sum_x > @mut_fre_x_thres['ispymac']
        """
        )
        .query("tem_0 > 0 or tem_1 > 0 or tem_2 > 0 or tem_3 > 0 or tem_4 > 0")
    )


def pairwise_transform(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = (
        df.assign(**{target: lambda df: df[target] / df["sum_x"]})
        .pivot(
            columns=["cas"],
            values=target,
            index="sgrna",
        )
        .reset_index(drop=False)
    )

    return df


def pairwise_test(
    df: pd.DataFrame,
    target: str,
    methods: list,
) -> None:
    a = df["spycas9"]
    result_dict = {"b": [], "method": [], "alternative": [], "p-value": []}
    for cas in ["spymac", "ispymac"]:
        for alternative in ["less", "greater"]:
            b = df[cas]

            #####################################
            # The t-test on independent samples (or the so-called unpaired t-test).
            #####################################

            if "t-test-ind" in methods:
                # equal_var=True performs normal unpaired t-test.
                # nan_policy="omit" omit missing data.
                result_dict["b"].append(cas)
                result_dict["method"].append("t-test-ind")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    ttest_ind(
                        a, b, equal_var=True, nan_policy="omit", alternative=alternative
                    ).pvalue.item()
                )

            if "Welch-test" in methods:
                # equal_var=False performs Welch’s t-test.
                result_dict["b"].append(cas)
                result_dict["method"].append("Welch-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    ttest_ind(
                        a,
                        b,
                        equal_var=False,
                        nan_policy="omit",
                        alternative=alternative,
                    ).pvalue.item()
                )

            if "Yuen-test" in methods:
                # trim=0.05 performs a trimmed (Yuen’s) t-test, which is an extension of Welch’s t-test and help to filter outliers.
                result_dict["b"].append(cas)
                result_dict["method"].append("Yuen-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    ttest_ind(
                        a, b, nan_policy="omit", alternative=alternative, trim=0.05
                    ).pvalue.item()
                )

            # # method=PermutationMethod(n_resamples=9999) performs permutation test over t-statistics.
            # result_dict["b"].append(cas)
            # result_dict["method"].append("permute-t-test")
            # result_dict["alternative"].append(alternative)
            # result_dict["p-value"].append(
            #     ttest_ind(
            #         a,
            #         b,
            #         nan_policy="omit",
            #         alternative=alternative,
            #         method=PermutationMethod(n_resamples=9999),
            #     ).pvalue.item()
            # )

            #####################################
            # The U-test on independent samples (or non-parametric version of the t-test for independent samples.)
            #####################################

            if "U-test" in methods:
                # The Mann-Whitney U test requires that the underlying distributions of two samples are the same. It does not care what the underlying distribution is.
                result_dict["b"].append(cas)
                result_dict["method"].append("U-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    mannwhitneyu(
                        a, b, alternative=alternative, nan_policy="omit"
                    ).pvalue.item()
                )

            # # method=PermutationMethod(n_resamples=9999) performs permutation test over U-statistics.
            # result_dict["b"].append(cas)
            # result_dict["method"].append("permute-U-test")
            # result_dict["alternative"].append(alternative)
            # result_dict["p-value"].append(
            #     mannwhitneyu(
            #         a,
            #         b,
            #         alternative=alternative,
            #         nan_policy="omit",
            #         method=PermutationMethod(n_resamples=9999),
            #     ).pvalue.item()
            # )

            ######################################
            # The binomial test.
            ######################################

            if "binomial-test" in methods:
                result_dict["b"].append(cas)
                result_dict["method"].append("binomial-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    binomtest(
                        (a > b).sum().item(),
                        ((a > b).sum().item() + (a < b).sum().item()),
                        p=0.5,
                        alternative=alternative,
                    ).pvalue.item()
                )

            #######################################
            # The t-test on two related samples (or the so-called paired t-test).
            #######################################

            if "t-test-rel" in methods:
                result_dict["b"].append(cas)
                result_dict["method"].append("t-test-rel")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    ttest_rel(
                        a, b, nan_policy="omit", alternative=alternative
                    ).pvalue.item()
                )

            ###########################################
            # The Wilcoxon signed-rank test (non-parametric version of the paired t-test).
            ###########################################

            if "Wilcoxon-signed-rank-test" in methods:
                result_dict["b"].append(cas)
                result_dict["method"].append("Wilcoxon-signed-rank-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    wilcoxon(
                        a,
                        b,
                        alternative=alternative,
                        method="auto",
                        nan_policy="omit",
                    ).pvalue.item()
                )

            # ############################################
            # # The Baumgartner-Weiss-Schindler test on two independent samples (cannot handle missing data).
            # ############################################
            # result_dict["b"].append(cas)
            # result_dict["method"].append("Baumgartner-Weiss-Schindler-test")
            # result_dict["alternative"].append(alternative)
            # bws_test(a, b, alternative=alternative)

            ################################################
            # The Wilcoxon rank-sum test for two samples. The null hypothesis that two sets of measurements are drawn from the same distribution.
            ################################################

            if "Wilcoxon-rank-sum-test" in methods:
                result_dict["b"].append(cas)
                result_dict["method"].append("Wilcoxon-rank-sum-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    ranksums(
                        a, b, alternative=alternative, nan_policy="omit"
                    ).pvalue.item()
                )

            #################################################
            # The Brunner-Munzel test on two independent samples. It does not require the equal variance (like the Wilcoxon-Mann-Whitney’s U-test,) and same distribution.
            #################################################

            if "Brunner-Munzel-test" in methods:
                result_dict["b"].append(cas)
                result_dict["method"].append("Brunner-Munzel-test")
                result_dict["alternative"].append(alternative)
                result_dict["p-value"].append(
                    brunnermunzel(
                        a, b, alternative=alternative, nan_policy="omit"
                    ).pvalue.item()
                )

    os.makedirs("result", exist_ok=True)
    result_df = (
        pd.DataFrame(result_dict)
        .pivot(columns="b", index=["method", "alternative"], values="p-value")
        .reset_index()
        .to_csv(f"result/{target}_pairwise_test.csv", index=False)
    )


def draw_box_bar(
    df: pd.DataFrame,
    target: str,
    yup: float,
):
    os.makedirs("result", exist_ok=True)

    df = (
        df.dropna()
        .query("spycas9 > 0 or spymac > 0 or ispymac > 0")
        .melt(
            id_vars="sgrna",
            value_vars=["spycas9", "spymac", "ispymac"],
            var_name="cas",
            value_name=target,
        )
    )

    (
        ggplot(
            df,
            mapping=aes(x="cas", y=target, fill="cas", color="cas"),
        )
        + geom_boxplot(outlier_alpha=0.0)
        + scale_x_discrete(limits=["spycas9", "spymac", "ispymac"])
        + scale_color_manual(
            values={
                "spycas9": "#FF0000",
                "spymac": "#00FF00",
                "ispymac": "#0000FF",
            }
        )
        + scale_fill_manual(
            values={
                "spycas9": "#FF0000",
                "spymac": "#00FF00",
                "ispymac": "#0000FF",
            }
        )
        + coord_cartesian(ylim=(0, yup))
    ).save(f"result/{target}_box.pdf")

    df = (
        df.groupby("cas")[target]
        .agg(["mean", "sem"])
        .reset_index(drop=False)
        .assign(
            up=lambda df: df["mean"] + df["sem"],
            low=lambda df: df["mean"] - df["sem"],
        )
    )

    (
        ggplot(
            df,
            mapping=aes(
                x="cas",
                y="mean",
                ymin="low",
                ymax="up",
                fill="cas",
                color="cas",
            ),
        )
        + geom_col()
        + geom_errorbar()
        + scale_x_discrete(limits=["spycas9", "spymac", "ispymac"])
        + scale_color_manual(
            values={
                "spycas9": "#FF0000",
                "spymac": "#00FF00",
                "ispymac": "#0000FF",
            }
        )
        + scale_fill_manual(
            values={
                "spycas9": "#FF0000",
                "spymac": "#00FF00",
                "ispymac": "#0000FF",
            }
        )
    ).save(f"result/{target}_bar.pdf")


def multinomial_test(row: pd.Series) -> float:
    row = row[[f"tem_{bp}" for bp in range(5)]]
    if sum(row) == 0:
        return 1.0

    chi2_statistic, p_value = chisquare(
        f_obs=row.to_list(), f_exp=[sum(row) / len(row)] * len(row)
    )
    return p_value


def umap_embed(df: pd.DataFrame) -> tuple[np.ndarray]:
    reducer = umap.UMAP()
    data = df[[f"tem_{bp}" for bp in range(5)]].to_numpy()
    scaled_data = StandardScaler().fit_transform(data / data.sum(axis=1, keepdims=True))
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
