from typing import Tuple

import pandas as pd


#  ae_1-runtime.csv
def load_class(
    model_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    one = pd.read_csv(f"{model_class}_1-runtime.csv")
    two = pd.read_csv(f"{model_class}_2-runtime.csv")
    three = pd.read_csv(f"{model_class}_3-runtime.csv")
    four = pd.read_csv(f"{model_class}_4-runtime.csv")
    five = pd.read_csv(f"{model_class}_5-runtime.csv")

    return one, two, three, four, five


# experiment,seed,model_class,family,tp,fp,tn,fn,precision,recall,f1,miner_time_sec,scientist_time_sec,at_time_sec,total_time_sec,model_wall_time_sec
def merge_dfs(
    one: pd.DataFrame,
    two: pd.DataFrame,
    three: pd.DataFrame,
    four: pd.DataFrame,
    five: pd.DataFrame,
) -> pd.DataFrame:
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
            "miner_time_sec": (
                one["miner_time_sec"][0]
                + two["miner_time_sec"][0]
                + three["miner_time_sec"][0]
                + four["miner_time_sec"][0]
                + five["miner_time_sec"][0]
            ),
            "scientist_time_sec": (
                one["scientist_time_sec"][0]
                + two["scientist_time_sec"][0]
                + three["scientist_time_sec"][0]
                + four["scientist_time_sec"][0]
                + five["scientist_time_sec"][0]
            ),
            "at_time_sec": (
                one["at_time_sec"][0]
                + two["at_time_sec"][0]
                + three["at_time_sec"][0]
                + four["at_time_sec"][0]
                + five["at_time_sec"][0]
            ),
            "total_time_sec": (
                one["total_time_sec"][0]
                + two["total_time_sec"][0]
                + three["total_time_sec"][0]
                + four["total_time_sec"][0]
                + five["total_time_sec"][0]
            ),
            "model_wall_time_sec": (
                one["model_wall_time_sec"][0]
                + two["model_wall_time_sec"][0]
                + three["model_wall_time_sec"][0]
                + four["model_wall_time_sec"][0]
                + five["model_wall_time_sec"][0]
            ),
        },
        index=[0],
    )

    return merged


if __name__ == "__main__":
    MODEL_CLASSES = ["ffnn", "cnn", "ae"]

    for model_class in MODEL_CLASSES:
        one, two, three, four, five = load_class(model_class)
        merged = merge_dfs(one, two, three, four, five)

        merged.to_csv(f"{model_class}_merged.csv", index=False)
