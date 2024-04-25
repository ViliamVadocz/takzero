import os
from numpy.random import choice
import matplotlib.pyplot as plt
from math import inf

BETA = 0.5


def load(path):
    with open(path) as file:
        lines = file.readlines()
    return [[m.split(":") for m in line.split(",")[:-1]] for line in lines]


def moves(xs):
    return [x[0] for x in xs]


def picked(xs):
    return max(xs, key=lambda x: int(x[1]))


def with_highest_visits(xs):
    max_visits = max(int(x[1]) for x in xs)
    return [x for x in xs if int(x[1]) == max_visits]


def sampled(xs):
    return choice(moves(xs), 1, [int(x[1]) for x in xs])[0]


def to_value(s: str):
    if "Win" in s:
        return 1
    if "Loss" in s:
        return -1
    if "Draw" in s:
        return 0
    return float(s)


def highest_value(xs):
    return max(-to_value(x[2]) for x in xs if int(x[1]) > 0)


def highest_value_plus_uncertainty(xs):
    return max(-to_value(x[2]) + BETA * float(x[3]) for x in xs if int(x[1]) > 0)


def choice_with_highest_value_plus_uncertainty(xs):
    return max(
        [x for x in xs if int(x[1]) > 0],
        key=lambda x: -to_value(x[2]) + BETA * float(x[3]),
    )


def picked_value(xs):
    ps = with_highest_visits(xs)
    return max(-to_value(p[2]) for p in ps)


def picked_value_plus_uncertainty(xs):
    p = picked(xs)
    return -to_value(p[2]) + BETA * float(p[3])


def how_many_picked_highest_uncertainty(xss):
    return sum(
        choice_with_highest_value_plus_uncertainty(xs) in with_highest_visits(xs)
        for xs in xss
    )


def index_of_highest_value_plus_uncertainty(xs):
    return sorted(
        xs,
        key=lambda x: int(x[1]),
        reverse=True,
    ).index(choice_with_highest_value_plus_uncertainty(xs))


# === === ===

runs = [(f[:-4], load(os.path.join("runs", f))) for f in os.listdir("runs")]
# runs = [
#     (name, run)
#     for name, run in runs
#     if name
#     in [
#         "puct",
#         "epuct",
#         "euct",
#         "seqhal_64_linear_euct",
#         "seqhal_64_linear_puct",
#         "seqhal_64_linear_epuct",
#     ]
# ]

m = [moves(xs) for xs in runs[0][1]]
for _name, run in runs:
    assert m == [moves(xs) for xs in run], run

print("how often was the most visited action the one that maximizes value+beta*std_dev")
for name, run in runs:
    print("-", name, ":", how_many_picked_highest_uncertainty(run))

print("what is the average index of the action that maximizes value+beta*std_dev")
for name, run in runs:
    print(
        "-",
        name,
        ":",
        sum(index_of_highest_value_plus_uncertainty(xs) for xs in run) / len(run),
    )

for name, run in runs:
    plt.plot(sorted(picked_value(xs) for xs in run), label=name)
plt.legend()
plt.grid()
plt.title("sorted picked value")
plt.show()

for name, run in runs:
    plt.plot(sorted(highest_value(xs) for xs in run), label=name)
plt.legend()
plt.grid()
plt.title("sorted highest value")
plt.show()

# for name, run in runs:
#     plt.plot(
#         [
#             t[1]
#             for t in sorted(
#                 enumerate(picked_value(xs) for xs in run),
#                 key=lambda t: picked_value(runs[0][1][t[0]]),
#             )
#         ],
#         label=name,
#     )
# plt.legend()
# plt.grid()
# plt.title("sorted picked value (sorted according to _)")
# plt.show()

for name, run in runs:
    plt.plot(
        sorted(picked_value_plus_uncertainty(xs) for xs in run),
        label=name,
    )
plt.legend()
plt.grid()
plt.title("sorted picked value+beta*std_dev")
plt.show()

for name, run in runs:
    plt.plot(sorted(highest_value_plus_uncertainty(xs) for xs in run), label=name)
plt.legend()
plt.grid()
plt.title("sorted highest value+beta*std_dev")
plt.show()

# picked
matrix = [[0] * len(runs) for _ in range(len(runs))]
picked_moves = [[picked(xs)[0] for xs in run] for _name, run in runs]
for i in range(len(picked_moves[0])):
    ms = [ms[i] for ms in picked_moves]
    for x, m1 in enumerate(ms):
        for y, m2 in enumerate(ms):
            if m1 == m2:
                matrix[x][y] += 1

names = [name for name, _ in runs]
print("," + ",".join(names))
for name, line in zip(names, matrix):
    print(name + "," + ",".join(map(str, line)))


# sampled
matrix = [[0] * len(runs) for _ in range(len(runs))]
for _ in range(200):
    picked_moves = [[sampled(xs) for xs in run] for _name, run in runs]
    for i in range(len(picked_moves[0])):
        ms = [ms[i] for ms in picked_moves]
        for x, m1 in enumerate(ms):
            for y, m2 in enumerate(ms):
                if m1 == m2:
                    matrix[x][y] += 1

names = [name for name, _ in runs]
print("," + ",".join(names))
for name, line in zip(names, matrix):
    print(name + "," + ",".join(map(str, line)))
