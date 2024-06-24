import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes


def read(path: str):
    with open(path, "r") as file:
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
        return steps, current, early, late, after


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (12, 6)})

if False:
    lcg = read("lcghash_data.csv")
    sim = read("simhash_data.csv")

    ax_sim: Axes
    ax_lcg: Axes
    fig, (ax_sim, ax_lcg) = plt.subplots(1, 2, sharey=True)

    ax_sim.set_title("SimHash")
    ax_sim.plot(sim[0], sim[1], linewidth=2)
    ax_sim.plot(sim[0], sim[2], linewidth=2)
    ax_sim.plot(sim[0], sim[3], linewidth=2)

    ax_lcg.set_title("LCGHash")
    ax_lcg.plot(lcg[0], lcg[1], linewidth=2, label="training")
    ax_lcg.plot(lcg[0], lcg[2], linewidth=2, label="early")
    ax_lcg.plot(lcg[0], lcg[3], linewidth=2, label="late")

    ax_lcg.legend()
    ax_sim.grid()
    ax_lcg.grid()
    ax_sim.set_ylim(0, 1)
    ax_lcg.set_ylim(0, 1)
    ax_sim.set_ylabel("Ratio")

    # fig.suptitle("Ratio of Positions Reported as \"Seen\" in the Batch")
    fig.supxlabel("Number of Batches Seen")

    plt.savefig("generalization_behaviour.png", bbox_inches="tight", dpi=160)

rnd = read("rnd_data.csv")
plt.plot(rnd[0], rnd[1], linewidth=2, label="training")
plt.plot(rnd[0], rnd[2], linewidth=2, label="early")
plt.plot(rnd[0], rnd[3], linewidth=2, label="late")
plt.plot(rnd[0], rnd[4], linewidth=2, label="seen")
plt.ylabel("Normalized RND output")
plt.xlabel("Training Batches")
plt.legend()
plt.grid()
plt.savefig("rnd_behaviour.png", bbox_inches="tight", dpi=160)
# plt.show()
