import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        'Parse command line arguments for summary-nightly-results script.')
    parser.add_argument('--results-folder',
                        type=str,
                        required=True,
                        help='The folder where the results are stored.')
    parser.add_argument('--description',
                        type=str,
                        required=True,
                        help='Description of the results.')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help="The dataset used for the benchmark.")

    args = parser.parse_args()
    return args

    
def get_perf(df, method, model, metric):
    
    means = []
    
    for qps in [2,4,8,16,"inf"]:
        target = df['Test name'].str.contains(model)
        target = target & df['Engine'].str.contains(method)
        target = target & df['Test name'].str.contains("qps_" + str(qps))
        filtered_df = df[target]

        if filtered_df.empty:
            means.append(0.)
        else:
            means.append(filtered_df[metric].values[0])

    return np.array(means)

def get_perf_w_std(df, method, model, metric):
    
    if metric in ["TTFT", "TPOT"]:
        mean = get_perf(df, method, model, "Mean " + metric + " (ms)")
        mean = mean.tolist()
        std = get_perf(df, method, model, "Std " + metric + " (ms)")
        if std.mean() == 0:
            std = None
        success = get_perf(df, method, model, "Successful req.")
        if std is not None:
            std = std / np.sqrt(success)
            std = std.tolist()

    else:
        assert metric == "Tput"
        mean = get_perf(df, method, model, "Input Tput (tok/s)") + get_perf(df, method, model, "Output Tput (tok/s)")
        mean = mean.tolist()
        std = None

    return mean, std


def main(args):
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())
            
    # generate markdown table
    df = pd.DataFrame.from_dict(results)
    df = df[df["Test name"].str.contains(args.dataset)]


    plt.rcParams.update({'font.size': 18})
    plt.set_cmap("cividis")

    # plot results
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(hspace=1)
    methods = ["vllm_055", "vllm_053post1"]
    for i, model in enumerate(["llama8B", "llama70B", "mixtral8x7B"]):
        for j, metric in enumerate(["Tput", "TPOT", "TTFT"]):
            
            ax = axes[i, j]
            if metric in ["TTFT", "TPOT"]:
                ax.set_ylabel(f"{metric} (ms)")
            else:
                ax.set_ylabel(f"Thoughput (tokens/s)")
            ax.set_xlabel("QPS")
            if metric == "Tput":
                ax.set_title(f"{model} {args.dataset} Thoughput")
            else:
                ax.set_title(f"{model} {args.dataset} {metric}")
            ax.grid(axis='y')
            print(model, metric)
            
            inf_qps_results = []
            for k, method in enumerate(methods):
                mean, std = get_perf_w_std(df, method, model, metric)
                ax.errorbar(range(len(mean)),
                            mean, 
                            yerr=std, 
                            capsize=10, 
                            capthick=4,
                            label="v0.5.5" if method == "vllm_055" else "v0.5.3.post1",
                            lw=4,)
                inf_qps_results.append(mean[-2])
            print((inf_qps_results[0] - inf_qps_results[1]) / inf_qps_results[1])
            ax.set_ylim(bottom=0)
            if metric == "TTFT":
                ax.set_ylim(0, 500)
            ax.set_xticks(range(len(mean)))
            ax.set_xticklabels(["2", "4", "8", "16", "inf"])

            ax.legend()


    fig.tight_layout()
    fig.savefig(f"nightly_results_{args.dataset}.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
