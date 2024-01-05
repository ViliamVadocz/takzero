import re
from pathlib import Path

# [2024-01-04T08:25:25Z INFO  evaluation] model_000500.ot vs. model_000300.ot: Evaluation { wins: 29, losses: 1, draws: 2 } 90.6%
PATTERN = re.compile(
    r"model_(\d\d\d\d\d\d)\.ot vs\. model_(\d\d\d\d\d\d)\.ot: Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
)

DATA_FILES = [
    "_data/eval-9219531.err",
    "_data/eval-9219618.err",
    "_data/eval-9219618.err",
    "_data/eval-9219618.err",
]
SAVE_FILE = "match_results.csv"
STEP = 100

# Clear file
with open(SAVE_FILE, "w"):
    pass

for path in DATA_FILES:
    with open(path, "r") as file:
        contents = file.read()
    with open(SAVE_FILE, "a") as file:
        file.writelines(
            f"{int(x[1]) - int(x[1]) % STEP}, {int(x[2]) - int(x[2]) % STEP}, {x[3]}, {x[4]}, {x[5]}\n"
            for x in re.finditer(PATTERN, contents)
        )
