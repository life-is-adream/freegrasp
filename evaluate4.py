from evaluate_common import evaluate_subset


def evaluate_pipeline():
    subset_name = "medium_yes_ambi"
    return evaluate_subset(subset_name)


if __name__ == "__main__":
    evaluate_pipeline()
