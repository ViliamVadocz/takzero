import csv
import matplotlib.pyplot as plt

with open("rnd_data.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # skip header
    rows = [
        (int(steps), float(x1), float(x2), float(x3)) for steps, x1, x2, x3 in reader
    ]
    steps = [s[0] for s in rows]
    loss = [s[1] for s in rows]
    early = [s[2] for s in rows]
    late = [s[3] for s in rows]


plt.plot(steps, loss, label="current")
plt.plot(steps, early, label="early")
plt.plot(steps, late, label="late")
plt.grid()
plt.legend()
plt.title("New Normalization")
plt.xlabel("training steps")
plt.ylabel("mean of normalized difference")
plt.ylim(bottom=0)
plt.show()
