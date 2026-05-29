from evaluate_common import evaluate_subset


def evaluate_pipeline():
    subset_names = [
        "easy_no_ambi",
        "easy_yes_ambi",
        "medium_no_ambi",
        "medium_yes_ambi",
        "hard_no_ambi",
        "hard_yes_ambi",
    ]

    scores = {}
    for subset_name in subset_names:
        scores[subset_name] = evaluate_subset(subset_name)

    return scores


if __name__ == "__main__":
    evaluate_pipeline()
