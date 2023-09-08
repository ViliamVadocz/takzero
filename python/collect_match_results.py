from pathlib import Path

RESULT_DIR = ".\\_data\\5x5\\1\\evaluation"

results = []

for file in Path(RESULT_DIR).iterdir():
    if not file.is_file() or not file.suffix == ".err":
        continue
    with open(file, "r") as file:
        lines = iter(file.readlines())
        next(lines)  # skip Begin

        while True:
            try:
                match = next(lines)
                result = next(lines)
            except StopIteration:
                break
            split = match.split()
            a, b = split[3], split[5]
            split = result.split()
            wins, losses, draws = split[6], split[8], split[10]

            results.append(f"{a[:6]}, {b[:6]}, {wins} {losses} {draws}\n")

with open("match_results.csv", "w") as file:
    file.writelines(results)
