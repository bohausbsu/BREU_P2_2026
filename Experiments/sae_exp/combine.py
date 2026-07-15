import pandas as pd


def load_batch(model_class, param_count):
    one = pd.read_csv(f"sae_{model_class}_{param_count}-1-0.3-0.02.csv", header=0)
    two = pd.read_csv(f"sae_{model_class}_{param_count}-2-0.3-0.02.csv", header=0)
    three = pd.read_csv(f"sae_{model_class}_{param_count}-3-0.3-0.02.csv", header=0)
    four = pd.read_csv(f"sae_{model_class}_{param_count}-4-0.3-0.02.csv", header=0)
    five = pd.read_csv(f"sae_{model_class}_{param_count}-5-0.3-0.02.csv", header=0)

    return one, two, three, four, five


def merge_dfs(
    one: pd.DataFrame,
    two: pd.DataFrame,
    three: pd.DataFrame,
    four: pd.DataFrame,
    five: pd.DataFrame,
):
    merged = pd.DataFrame(
        {
            "model_class": one["model_class"][0],
            "family": one["family"][0],
            "tp": (
                one["tp"][0]
                + two["tp"][0]
                + three["tp"][0]
                + four["tp"][0]
                + five["tp"][0]
            )
            / 5,
            "fp": (
                one["fp"][0]
                + two["fp"][0]
                + three["fp"][0]
                + four["fp"][0]
                + five["fp"][0]
            )
            / 5,
            "tn": (
                one["tn"][0]
                + two["tn"][0]
                + three["tn"][0]
                + four["tn"][0]
                + five["tn"][0]
            )
            / 5,
            "fn": (
                one["fn"][0]
                + two["fn"][0]
                + three["fn"][0]
                + four["fn"][0]
                + five["fn"][0]
            )
            / 5,
            "precision": (
                one["precision"][0]
                + two["precision"][0]
                + three["precision"][0]
                + four["precision"][0]
                + five["precision"][0]
            )
            / 5,
            "recall": (
                one["recall"][0]
                + two["recall"][0]
                + three["recall"][0]
                + four["recall"][0]
                + five["recall"][0]
            )
            / 5,
            "f1": (
                one["f1"][0]
                + two["f1"][0]
                + three["f1"][0]
                + four["f1"][0]
                + five["f1"][0]
            )
            / 5,
            "miner_mem_bytes": (
                one["miner_mem_bytes"][0]
                + two["miner_mem_bytes"][0]
                + three["miner_mem_bytes"][0]
                + four["miner_mem_bytes"][0]
                + five["miner_mem_bytes"][0]
            )
            / 5,
            "miner_mem_mb": (
                one["miner_mem_mb"][0]
                + two["miner_mem_mb"][0]
                + three["miner_mem_mb"][0]
                + four["miner_mem_mb"][0]
                + five["miner_mem_mb"][0]
            )
            / 5,
            "scientist_mem_bytes": (
                one["scientist_mem_bytes"][0]
                + two["scientist_mem_bytes"][0]
                + three["scientist_mem_bytes"][0]
                + four["scientist_mem_bytes"][0]
                + five["scientist_mem_bytes"][0]
            )
            / 5,
            "scientist_mem_mb": (
                one["scientist_mem_mb"][0]
                + two["scientist_mem_mb"][0]
                + three["scientist_mem_mb"][0]
                + four["scientist_mem_mb"][0]
                + five["scientist_mem_mb"][0]
            )
            / 5,
            "at_mem_bytes": (
                one["at_mem_bytes"][0]
                + two["at_mem_bytes"][0]
                + three["at_mem_bytes"][0]
                + four["at_mem_bytes"][0]
                + five["at_mem_bytes"][0]
            )
            / 5,
            "at_mem_mb": (
                one["at_mem_mb"][0]
                + two["at_mem_mb"][0]
                + three["at_mem_mb"][0]
                + four["at_mem_mb"][0]
                + five["at_mem_mb"][0]
            )
            / 5,
            "total_mem_bytes": (
                one["total_mem_bytes"][0]
                + two["total_mem_bytes"][0]
                + three["total_mem_bytes"][0]
                + four["total_mem_bytes"][0]
                + five["total_mem_bytes"][0]
            )
            / 5,
            "total_mem_mb": (
                one["total_mem_mb"][0]
                + two["total_mem_mb"][0]
                + three["total_mem_mb"][0]
                + four["total_mem_mb"][0]
                + five["total_mem_mb"][0]
            )
            / 5,
        },
        index=[0],
    )

    return merged


if __name__ == "__main__":
    MODEL_CLASSES = ["ae", "cnn", "ffnn"]
    PARAM_COUNTS = ["4k", "6k", "8k", "10k", "12k"]

    for model_class in MODEL_CLASSES:
        for param_count in PARAM_COUNTS:
            one, two, three, four, five = load_batch(model_class, param_count)
            merged = merge_dfs(one, two, three, four, five)

            merged.to_csv(f"sae_{model_class}_{param_count}_merged.csv", index=False)
