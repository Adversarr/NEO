"""
Input:

xxx-yyy,UID
...

Output

UID
...
"""

import csv
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default="kiuisobj_v1_merged_80K.csv", help="Path to the UID file.")
    parser.add_argument("--output-file", type=str, default="outputs/kiuisobj_v1_merged_80K.txt", help="Path to the output file.")
    return parser.parse_args()

def main(args):
    with open(args.input_file, "r") as f:
        reader = csv.reader(f)
        uids = [row[1] for row in reader]
    with open(args.output_file, "w") as f:
        for uid in uids:
            f.write(uid + "\n")

    print(f"Filtered {len(uids)} UIDs to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)



