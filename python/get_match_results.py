import re
from pathlib import Path

PATTERN = re.compile(r"(\d\d\d\d\d\d), (\d\d\d\d\d\d), (\d*), (\d*), (\d*)")

DATA_FILE = "match_results.csv"
SAVE_FILE = "match_results_1.csv"
STEP = 1000

with open(DATA_FILE, "r") as file:
    contents = file.read()
with open(SAVE_FILE, "a") as file:
    file.writelines(
        f"{int(x[1]) - int(x[1]) % STEP}, {int(x[2]) - int(x[2]) % STEP}, {x[3]}, {x[4]}, {x[5]}\n"
        for x in re.finditer(PATTERN, contents)
    )
