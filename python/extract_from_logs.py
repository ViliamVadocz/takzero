import re
import os
import matplotlib.pyplot as plt
import numpy as np

POLICY_PATTERN = re.compile(r"loss_policy = \[(\d+\.\d+)\]")
VALUE_PATTERN = re.compile(r"loss_value = \[(\d+\.\d+)\]")
UBE_PATTERN = re.compile(r"loss_ube = \[(\d+\.\d+)\]")
RND_PATTERN = re.compile(r"loss_rnd = \[(\d+\.\d+)\]")

UBE_STATS_PATTERN = re.compile(
    r"\[UBE STATS\] ply: (\d+), bf: (\d+), root: (\d+\.\d+), max: (\d+\.\d+), target: (\d+\.\d+)"
)
# UBE_STATS_PATTERN = re.compile(
#     r"\[UBE STATS\] ply: (\d+), root: (\d+\.\d+), max: (\d+\.\d+), selected: (\d+\.\d+)"
# )


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

# data_per_step = dict()
# directory = "_data"
# for filename in os.listdir(directory):
#     if "reanalyze" not in filename:
#         continue
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f):
#         with open(f) as file:
#             contents = file.read()
#         steps = contents.split("Number of positions:")
#         for i, step in enumerate(steps[1:]):
#             ube_stats = [
#                 (int(x[1]), float(x[2]), float(x[3]), float(x[4]))
#                 for x in re.finditer(UBE_STATS_PATTERN, step)
#             ]
#             data = data_per_step.setdefault(i, [])
#             data += ube_stats
# data_per_step = list(data_per_step.values())
# print(len(data_per_step))

# ube_per_bf = dict()
# for data in data_per_step:
#     for tup in data:
#         l = ube_per_bf.setdefault(tup[1], [])
#         l.append(tup[2])
# ube_per_bf = sorted(ube_per_bf.items())
# plt.plot(
#     [x for x, y in ube_per_bf],
#     [np.mean(y) for x, y in ube_per_bf],
# )
# plt.xlabel("Branching factor")
# plt.ylabel("Root UBE")
# plt.title("Branching factor vs root UBE")
# plt.ylim(0, 1)

# bf_per_ply = dict()
# for data in data_per_step:
#     for tup in data:
#         l = bf_per_ply.setdefault(tup[0], [])
#         l.append(tup[1])
# bf_per_ply = sorted(bf_per_ply.items())
# print(bf_per_ply[0][0])
# plt.plot(
#     [x for x, y in bf_per_ply][::2],
#     [np.mean(y) for x, y in bf_per_ply][::2],
#     label="black",
# )
# plt.plot(
#     [x for x, y in bf_per_ply][1::2],
#     [np.mean(y) for x, y in bf_per_ply][1::2],
#     label="white",
# )
# plt.xlabel("Game ply")
# plt.ylabel("Branching factor")
# plt.title("Game ply vs branching factor")

# n = 1
# step_size = len(data_per_step) / n
# for i in range(n):
#     start = int(step_size * i)
#     end = int(step_size * (i + 1))
#     ube_per_ply = {i: [] for i in range(150)}
#     for data in data_per_step[start:end]:
#         for tup in data:
#             ube_per_ply[tup[0]].append(tup[2])
#     root = [
#         (i, np.mean(ube), np.std(ube)) for i, ube in ube_per_ply.items() if len(ube) > 0
#     ]
#     plt.plot([x for x, _, _ in root], [y for _, y, _ in root], label=f"[{start},{end})")
#     plt.fill_between(
#         [x for x, _, _ in root],
#         [y - z for _, y, z in root],
#         [y + z for _, y, z in root],
#         alpha=0.2,
#     )

# plt.xlabel("Game plies (half-moves)")
# plt.ylabel("Mean root UBE")
# plt.title("Root UBE throughout games for several reanalyze step ranges")
# plt.ylim(0, 1)


# def mean_and_std(data, ply_step, i):
#     l = [tup[2] for tup in data if i * ply_step <= tup[0] < (i + 1) * ply_step]
#     if len(l) == 0:
#         return None, None
#     return (np.mean(l), np.std(l))


# ply_step = 20
# for i in range(4):
#     root = [mean_and_std(data, ply_step, i) for data in data_per_step]
#     ube = moving_average([a for a, _ in root], 100)
#     std = moving_average([b for _, b in root], 100)
#     plt.plot(
#         ube,
#         label=f"[{i * ply_step},{(i + 1) * ply_step})",
#     )
#     plt.fill_between(
#         list(range(len(ube))),
#         [a - b for a, b in zip(ube, std)],
#         [a + b for a, b in zip(ube, std)],
#         alpha=0.2,
#     )

# plt.xlabel("reanalyze steps")
# plt.ylabel("Root UBE (Moving average size 100)")
# plt.title("Root UBE during reanalyze, for several game ply (half-move) ranges")
# plt.ylim(0, 1)

# plt.hist(
#     [tup[0] for data in data_per_step for tup in data],
#     density=True,
#     bins=list(range(120)),
# )
# plt.xlabel("Game ply (half-move)")
# plt.ylabel("Density")

with open("_data\\replays.txt", "r") as file:
    game_lengths = [
        len(line.split("]")[1].strip().split()) + 1
        for line in file.readlines()
        if len(line) > 2
    ]

# plt.hist(game_lengths, bins=[x for x in range(130)], density=True)
# plt.xlabel("Game length in plies (half-moves)")
# plt.ylabel("Density")

plt.plot(moving_average(game_lengths, 256))
plt.ylabel("Moving average of game length (size = 256)")
plt.xlabel("Replays")
plt.title("Game length during self-play")
plt.ylim(0)

# plt.legend()
plt.show()
