from numpy.random import choice
import matplotlib.pyplot as plt

BETA = 0.5


def load(path):
    with open(path) as file:
        lines = file.readlines()
    return [[m.split(":") for m in line.split(",")[:-1]] for line in lines]


def picked(xs):
    return max(xs, key=lambda x: int(x[1]))[0]


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
        key=lambda x: to_value(x[2]) + BETA * float(x[3]),
    )[0]


def how_many_picked_highest_uncertainty(xss):
    return sum(
        choice_with_highest_value_plus_uncertainty(xs) == picked(xs) for xs in xss
    )


baseline = load("runs/baseline.txt")
beta_uniform = load("runs/beta_uniform.txt")
top_5 = load("runs/constrained_top_5_with_beta.txt")
top_8_no_uniform = load("runs/constrained_top_8_with_beta_no_uniform.txt")

print("baseline", how_many_picked_highest_uncertainty(baseline))
print("beta_uniform", how_many_picked_highest_uncertainty(beta_uniform))
print("top_5", how_many_picked_highest_uncertainty(top_5))
print("top_8_no_uniform", how_many_picked_highest_uncertainty(top_8_no_uniform))

plt.hist(
    [highest_value_plus_uncertainty(xs) for xs in baseline],
    label="baseline",
    bins=[i / 25 for i in range(50)],
    alpha=0.8,
)
plt.hist(
    [highest_value_plus_uncertainty(xs) for xs in beta_uniform],
    label="beta-uniform",
    bins=[i / 25 for i in range(50)],
    alpha=0.8,
)
plt.hist(
    [highest_value_plus_uncertainty(xs) for xs in top_5],
    label="top5",
    bins=[i / 25 for i in range(50)],
    alpha=0.8,
)
plt.hist(
    [highest_value_plus_uncertainty(xs) for xs in top_8_no_uniform],
    label="top8-no-uniform",
    bins=[i / 25 for i in range(50)],
    alpha=0.8,
)
plt.legend()
plt.grid()
plt.ylabel("value + beta * uncertainty")
plt.show()

same = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for base, beta_uni, t5, t8_no_uni in zip(
    baseline, beta_uniform, top_5, top_8_no_uniform
):
    assert len(base) == len(beta_uni) == len(t5) == len(t8_no_uni)

    for _ in range(100):
        picks = [sampled(base), sampled(beta_uni), sampled(t5), sampled(t8_no_uni)]
        for x, a in enumerate(picks):
            for y, b in enumerate(picks):
                if a == b:
                    same[x][y] += 1

print(same)
