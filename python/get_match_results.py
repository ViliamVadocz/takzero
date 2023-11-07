import re
from pathlib import Path

PATTERN = re.compile(
    r"\[.* evaluation\] (\d\d\d\d\d\d)_steps\.ot vs\. (\d\d\d\d\d\d)_steps.ot\n\[.* evaluation\] Evaluation { wins: (\d*), losses: (\d*), draws: (\d*) }"
)

SAVE_FILE = "match_results_1.csv"
STEP = 5000

with open(SAVE_FILE, "w"):
    pass
for path in Path("./_data/eval1").iterdir():
    with open(path, "r") as file:
        contents = file.read()
    with open(SAVE_FILE, "a") as file:
        file.writelines(
            f"{int(x[1]) - int(x[1]) % STEP}, {int(x[2]) - int(x[2]) % STEP}, {x[3]}, {x[4]}, {x[5]}\n"
            for x in re.finditer(PATTERN, contents)
        )
