import csv
import matplotlib.pyplot as plt

with open("rnd_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    rows = [
        (int(steps), float(x1), float(x2), float(x3), float(x4), float(x5))
        for steps, x1, x2, x3, x4, x5 in reader
    ]
    steps = [s[0] for s in rows]
    loss = [s[1] for s in rows]
    early = [s[2] for s in rows]
    late = [s[3] for s in rows]
    batch5k = [s[4] for s in rows]
    batch20k = [s[5] for s in rows]


plt.plot(steps, loss, label="current")
plt.plot(steps, early, label="early (4 ply)")
plt.plot(steps, late, label="late (120 ply)")
plt.plot(steps, late, label="batch at 5k")
plt.plot(steps, late, label="batch at 20k")
plt.grid()
plt.legend()
plt.title("RND output")
plt.xlabel("training steps")
plt.ylabel("mean of squared difference")
plt.ylim(bottom=0)
plt.show()
