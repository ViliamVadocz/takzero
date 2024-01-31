import re
import os
import matplotlib.pyplot as plt
import numpy as np

POLICY_PATTERN = re.compile(r"loss_policy = \[(\d+\.\d+)\]")
VALUE_PATTERN = re.compile(r"loss_value = \[(\d+\.\d+)\]")
UBE_PATTERN = re.compile(r"loss_ube = \[(\d+\.\d+)\]")
RND_PATTERN = re.compile(r"loss_rnd = \[(\d+\.\d+)\]")

UBE_STATS_PATTERN = re.compile(
    r"\[UBE STATS\] ply: (\d+), root: (\d+\.\d+), max: (\d+\.\d+), selected: (\d+\.\d+)"
)


def moving_average(a, n=3):
    l = [x for x in a if x is not None]
    assert len(l) != 0
    ret = np.cumsum(l, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# with open("_data/learn-9271401.err", "r") as file:
#     contents = file.read()
# policy_loss = list(float(x[1]) for x in re.finditer(POLICY_PATTERN, contents))
# value_loss = list(float(x[1]) for x in re.finditer(VALUE_PATTERN, contents))
# ube_loss = list(float(x[1]) for x in re.finditer(UBE_PATTERN, contents))
# rnd_loss = list(float(x[1]) for x in re.finditer(RND_PATTERN, contents))

# num_steps = len(ube_loss)
# steps = list(range(num_steps))

# plt.title("RND loss")
# plt.xlabel("training steps")
# plt.ylabel("loss")
# plt.plot(steps, rnd_loss, label="Raw batch loss")
# plt.plot(
#     steps[64:-63], moving_average(rnd_loss, 128), label="Moving average of size 128"
# )
# plt.ylim(0, 10)

# Get UBE stats from self-play.
data_per_step = dict()
directory = "_data"
for filename in os.listdir(directory):
    if "selfplay" not in filename:
        continue
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        with open(f) as file:
            contents = file.read()
        steps = contents.split("Step:")
        for i, step in enumerate(steps[1:]):
            ube_stats = [
                (int(x[1]), float(x[2]), float(x[3]), float(x[4]))
                for x in re.finditer(UBE_STATS_PATTERN, step)
            ]
            data = data_per_step.setdefault(i, [])
            data += ube_stats
data_per_step = list(data_per_step.values())

self_play_steps = len(data_per_step)

n = 4
for i in range(n):
    step_size = self_play_steps / n
    start = int(step_size * i)
    end = int(step_size * (i + 1))
    ube_per_ply = {i: [] for i in range(150)}
    for data in data_per_step[start:end]:
        for tup in data:
            ube_per_ply[tup[0]].append(tup[1])
    root = [(i, np.mean(ube)) for i, ube in ube_per_ply.items() if len(ube) > 0][:80]
    plt.plot([x for x, y in root], [y for x, y in root], label=f"[{start},{end})")

plt.xlabel("Game plies (half-moves)")
plt.ylabel("Mean root UBE")
plt.title("Root UBE throughout games for several training step ranges")
plt.ylim(0, 1)


# def mean(data, ply_step, i):
#     l = [tup[1] for tup in data if i * ply_step <= tup[0] < (i + 1) * ply_step]
#     if len(l) == 0:
#         return None
#     return np.mean(l)

# ply_step = 20
# for i in range(4):
#     root = [mean(data, ply_step, i) for data in data_per_step]
#     plt.plot(moving_average(root, 100), label=f"[{i * ply_step},{(i + 1) * ply_step})")

# plt.xlabel("Self-play steps")
# plt.ylabel("Root UBE (Moving average size 100)")
# plt.title("Root UBE during self-play, for several game ply (half-move) ranges")
# plt.ylim(0, 1)

# plt.hist(
#     [tup[0] for data in data_per_step for tup in data],
#     density=True,
#     bins=list(range(120)),
# )
# plt.xlabel("Game ply (half-move)")
# plt.ylabel("Density")

plt.legend()
plt.show()
