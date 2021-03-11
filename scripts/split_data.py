from sklearn.model_selection import train_test_split
import pandas as pd
import os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--csv_path", default="../data/HIV.csv")
    parser.add_argument("--label_name", default="HIV_active")

    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    train, test = train_test_split(
        data, stratify=data[args.label_name], random_state=42, test_size=0.2
    )

    train.to_csv(os.path.splitext(args.csv_path)[0] + "_train.csv", index=False)
    test.to_csv(os.path.splitext(args.csv_path)[0] + "_test.csv", index=False)
