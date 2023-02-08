import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot_settings import *


def test_acc(fpath):
    df = pd.read_csv(fpath)
    return df.test_acc.iloc[-1]


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    # s_shift
    unimodal_means, unimodal_sds = [], []
    multimodal_means, multimodal_sds = [], []
    for beta in args.beta_range:
        unimodal_values, multimodal_values = [], []
        for seed in range(args.n_seeds):
            unimodal_fpath = os.path.join(args.dpath, f"beta={beta}", "unimodal", f"version_{seed}", "metrics.csv")
            multimodal_fpath = os.path.join(args.dpath, f"beta={beta}", "multimodal", f"version_{seed}", "metrics.csv")
            unimodal_values.append(test_acc(unimodal_fpath))
            multimodal_values.append(test_acc(multimodal_fpath))
        unimodal_means.append(np.mean(unimodal_values))
        unimodal_sds.append(np.std(unimodal_values))
        multimodal_means.append(np.mean(multimodal_values))
        multimodal_sds.append(np.std(multimodal_values))
    ax.errorbar(range(len(args.beta_range)), unimodal_means, unimodal_sds, label="Unimodal ensemble")
    ax.errorbar(np.arange(len(args.beta_range)) + 0.05, multimodal_means, multimodal_sds, label="Multidimensional")
    ax.set_xticks(range(len(args.beta_range)))
    ax.set_xticklabels(args.beta_range)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Accuracy")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs("fig", exist_ok=True)
    plt.savefig(os.path.join("fig", "acc.png"), bbox_inches="tight")
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dpath", type=str, default="results")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--beta_range", nargs="+", type=int, default=[-4, -2, 0, 2, 4])
    main(parser.parse_args())