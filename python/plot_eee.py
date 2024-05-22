import csv
import matplotlib.pyplot as plt

with open("eee_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    rows = [
        (
            int(steps),
            float(x1),
            float(x2),
            float(x3),
            float(x4),
            float(x5),
            float(x6),
            float(x7),
        )
        for steps, x1, x2, x3, x4, x5, x6, x7 in reader
    ]
    steps = [s[0] for s in rows]
    current = [s[1] for s in rows]
    after = [s[2] for s in rows]
    early = [s[3] for s in rows]
    late = [s[4] for s in rows]
    random_early = [s[5] for s in rows]
    random_late = [s[6] for s in rows]
    impossible_early = [s[7] for s in rows]


plt.plot(steps, current, label="current")
plt.plot(steps, after, label="after")
plt.plot(steps, early, label="early (8 plies)")
plt.plot(steps, late, label="late (60 plies)")
plt.plot(steps, random_early, label="random early")
plt.plot(steps, random_late, label="random late")
plt.plot(steps, impossible_early, label="impossible early")
plt.grid()
plt.legend()
plt.title("Naive RND")
plt.xlabel("training steps")
plt.ylabel("signal")
plt.ylim(bottom=0)
plt.savefig("eee.svg")
plt.show()
