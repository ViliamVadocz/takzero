import re
import sys
from pathlib import Path

PATTERN = re.compile(
    # r"model_(\d\d\d\d\d\d)\.ot vs\. model_(\d\d\d\d\d\d)\.ot: Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
    r"([\w_-]+)_(\d+)\.ot vs\. ([\w_-]+)_(\d+)\.ot: Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
    # r"(\w*) vs. (\w*): Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
)

SAVE_FILE = "match_results.csv"
# SAVE_FILE = "match_results_tournament.csv"

# Clear file
with open(SAVE_FILE, "w"):
    pass

files = Path(sys.argv[1]).glob("eval-*.err")
for path in files:
    with open(path, "r") as file:
        contents = file.read()
    with open(SAVE_FILE, "a") as file:
        file.writelines(
            # f"{int(x[1])}, {int(x[2])}, {x[3]}, {x[4]}, {x[5]}\n"
            f"{x[1]}, {int(x[2])}, {x[3]}, {int(x[4])}, {x[5]}, {x[6]}, {x[7]}\n"
            # f"{x[1]}, {x[2]}, {x[3]}, {x[4]}, {x[5]}\n"
            for x in re.finditer(PATTERN, contents)
        )
