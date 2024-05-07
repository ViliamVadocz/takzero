import csv
import matplotlib.pyplot as plt

with open("rnd_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    rows = [
        (int(steps), float(x1), float(x2), float(x3), float(x4), float(x5), float(x6))
        for steps, x1, x2, x3, x4, x5, x6 in reader
    ]
    steps = [s[0] for s in rows]
    loss = [s[1] for s in rows]
    early = [s[2] for s in rows]
    late = [s[3] for s in rows]
    random_early = [s[4] for s in rows]
    random_late = [s[5] for s in rows]
    impossible_early = [s[6] for s in rows]


plt.plot(steps, loss, label="current")
plt.plot(steps, early, label="early")
plt.plot(steps, late, label="late")
plt.plot(steps, random_early, label="random early (8 ply)")
plt.plot(steps, random_late, label="random late (120 ply)")
plt.plot(steps, impossible_early, label="impossible early")
plt.grid()
plt.legend()
plt.title("RND output")
plt.xlabel("training steps")
plt.ylabel("mean of squared difference")
plt.ylim(bottom=0)
plt.show()
