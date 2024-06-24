import re
import sys
from pathlib import Path

# [2024-06-23T18:22:07Z INFO  evaluation] lcghash_low_beta_00-0200000.ot vs. simhash_mid_beta_02-0000000.ot: Evaluation { wins: 64, losses: 0, draws: 0 } 100.0%
PATTERN = re.compile(
    # r"model_(\d\d\d\d\d\d)\.ot vs\. model_(\d\d\d\d\d\d)\.ot: Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
    r"([\w\_\-]+)[\_\-](\d+)\.ot vs\. ([\w\d_\-]+)[\_\-](\d+)\.ot: Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
    # r"(\w*) vs. (\w*): Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
)

SAVE_FILE = "match_results.csv"
# SAVE_FILE = "match_results_tournament.csv"

# Clear file
with open(SAVE_FILE, "w"):
    pass

files = Path(sys.argv[1]).glob("eval-*.err")
for path in files:
    print(path)
    with open(path, "r") as file:
        eval_directory = file.readline()
        contents = file.read()
        if eval_directory.split("/")[0] == "runs":
            model_name = eval_directory.split("/")[1].replace("_", "-")[4:-1]
            print(model_name)
            contents = contents.replace("model", model_name)
    with open(SAVE_FILE, "a") as file:
        results = [
            # f"{int(x[1])}, {int(x[2])}, {x[3]}, {x[4]}, {x[5]}\n"
            f"{x[1]}, {int(x[2])}, {x[3]}, {int(x[4])}, {x[5]}, {x[6]}, {x[7]}\n"
            # f"{x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}\n"
            for x in re.finditer(PATTERN, contents)
        ]
        print(len(results))
        file.writelines(results)
