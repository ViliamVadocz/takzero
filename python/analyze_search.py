import os
from numpy.random import choice
import matplotlib.pyplot as plt

BETA = 0.0


def load(path):
    with open(path) as file:
        lines = file.readlines()
    return [[m.split(":") for m in line.split(",")[:-1]] for line in lines]


def picked(xs):
    return max(xs, key=lambda x: int(x[1]))


def sampled(xs):
    return choice([x[0] for x in xs], 1, [int(x[1]) for x in xs])


def to_value(s: str):
    if "Win" in s:
        return 1
    if "Loss" in s:
        return -1
    if "Draw" in s:
        return 0
    return float(s)


def highest_value_plus_uncertainty(xs):
    return max(to_value(x[2]) + BETA * float(x[3]) for x in xs)


def choice_with_highest_value_plus_uncertainty(xs):
    return max(
        [x for x in xs if int(x[1]) > 0],
        key=lambda x: -to_value(x[2]) + BETA * float(x[3]),
    )


def picked_value_plus_uncertainty(xs):
    p = picked(xs)
    return -to_value(p[2]) + BETA * float(p[3])


def how_many_picked_highest_uncertainty(xss):
    return sum(
        choice_with_highest_value_plus_uncertainty(xs) == picked(xs) for xs in xss
    )


runs = [(f[:-4], load(os.path.join("runs", f))) for f in os.listdir("runs")]

print("how often was the most visited action the one that maximizes value+beta*std_dev")
for name, run in runs:
    print("-", name, ":", how_many_picked_highest_uncertainty(run))

for name, run in runs:
    plt.plot(sorted(highest_value_plus_uncertainty(xs) for xs in run), label=name)
plt.legend()
plt.grid()
plt.title("sorted highest value")
plt.show()

for name, run in runs:
    plt.plot(
        sorted(picked_value_plus_uncertainty(xs) for xs in run),
        label=name,
    )
plt.legend()
plt.grid()
plt.title("sorted picked value+beta*std_dev")
plt.show()

picked_val = [
    (name, [picked_value_plus_uncertainty(xs) for xs in run]) for name, run in runs
]
for name, run in picked_val:
    plt.plot(
        [k for _, k in sorted(enumerate(run), key=lambda t: picked_val[7][1][t[0]])],
        label=name,
    )
plt.legend()
plt.grid()
plt.title("sorted (according to baseline) picked value+beta*std_dev")
plt.show()

[0]
