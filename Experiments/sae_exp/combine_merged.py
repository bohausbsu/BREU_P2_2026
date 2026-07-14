import pandas as pd


def load_class(model_class):
    four = pd.read_csv(f"sae_{model_class}_4k_merged.csv", header=0)
    six = pd.read_csv(f"sae_{model_class}_6k_merged.csv", header=0)
    eight = pd.read_csv(f"sae_{model_class}_8k_merged.csv", header=0)
    ten = pd.read_csv(f"sae_{model_class}_10k_merged.csv", header=0)
    twelve = pd.read_csv(f"sae_{model_class}_12k_merged.csv", header=0)

    return four, six, eight, ten, twelve


def merge_dfs(
    four: pd.DataFrame,
    six: pd.DataFrame,
    eight: pd.DataFrame,
    ten: pd.DataFrame,
    twelve: pd.DataFrame,
):
    merged = pd.DataFrame(
        {
            "model_class": four["model_class"][0],
            "family": four["family"][0],
            "tp": (
                four["tp"][0]
                + six["tp"][0]
                + eight["tp"][0]
                + ten["tp"][0]
                + twelve["tp"][0]
            )
            / 5,
            "fp": (
                four["fp"][0]
                + six["fp"][0]
                + eight["fp"][0]
                + ten["fp"][0]
                + twelve["fp"][0]
            )
            / 5,
            "tn": (
                four["tn"][0]
                + six["tn"][0]
                + eight["tn"][0]
                + ten["tn"][0]
                + twelve["tn"][0]
            )
            / 5,
            "fn": (
                four["fn"][0]
                + six["fn"][0]
                + eight["fn"][0]
                + ten["fn"][0]
                + twelve["fn"][0]
            )
            / 5,
            "precision": (
                four["precision"][0]
                + six["precision"][0]
                + eight["precision"][0]
                + ten["precision"][0]
                + twelve["precision"][0]
            )
            / 5,
            "recall": (
                four["recall"][0]
                + six["recall"][0]
                + eight["recall"][0]
                + ten["recall"][0]
                + twelve["recall"][0]
            )
            / 5,
            "f1": (
                four["f1"][0]
                + six["f1"][0]
                + eight["f1"][0]
                + ten["f1"][0]
                + twelve["f1"][0]
            )
            / 5,
            "miner_mem_bytes": (
                four["miner_mem_bytes"][0]
                + six["miner_mem_bytes"][0]
                + eight["miner_mem_bytes"][0]
                + ten["miner_mem_bytes"][0]
                + twelve["miner_mem_bytes"][0]
            )
            / 5,
            "miner_mem_mb": (
                four["miner_mem_mb"][0]
                + six["miner_mem_mb"][0]
                + eight["miner_mem_mb"][0]
                + ten["miner_mem_mb"][0]
                + twelve["miner_mem_mb"][0]
            )
            / 5,
            "scientist_mem_bytes": (
                four["scientist_mem_bytes"][0]
                + six["scientist_mem_bytes"][0]
                + eight["scientist_mem_bytes"][0]
                + ten["scientist_mem_bytes"][0]
                + twelve["scientist_mem_bytes"][0]
            )
            / 5,
            "scientist_mem_mb": (
                four["scientist_mem_mb"][0]
                + six["scientist_mem_mb"][0]
                + eight["scientist_mem_mb"][0]
                + ten["scientist_mem_mb"][0]
                + twelve["scientist_mem_mb"][0]
            )
            / 5,
            "at_mem_bytes": (
                four["at_mem_bytes"][0]
                + six["at_mem_bytes"][0]
                + eight["at_mem_bytes"][0]
                + ten["at_mem_bytes"][0]
                + twelve["at_mem_bytes"][0]
            )
            / 5,
            "at_mem_mb": (
                four["at_mem_mb"][0]
                + six["at_mem_mb"][0]
                + eight["at_mem_mb"][0]
                + ten["at_mem_mb"][0]
                + twelve["at_mem_mb"][0]
            )
            / 5,
            "total_mem_bytes": (
                four["total_mem_bytes"][0]
                + six["total_mem_bytes"][0]
                + eight["total_mem_bytes"][0]
                + ten["total_mem_bytes"][0]
                + twelve["total_mem_bytes"][0]
            )
            / 5,
            "total_mem_mb": (
                four["total_mem_mb"][0]
                + six["total_mem_mb"][0]
                + eight["total_mem_mb"][0]
                + ten["total_mem_mb"][0]
                + twelve["total_mem_mb"][0]
            )
            / 5,
        },
        index=[0],
    )

    return merged


if __name__ == "__main__":
    MODEL_CLASSES = ["ae", "cnn", "ffnn"]

    for model_class in MODEL_CLASSES:
        four, six, eight, ten, twelve = load_class(model_class)
        merged = merge_dfs(four, six, eight, ten, twelve)

        merged.to_csv(f"sae_{model_class}_total_merged.csv", index=False)
