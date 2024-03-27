from numpy.random import choice


def load(path):
    with open(path) as file:
        lines = file.readlines()
    return [
        [(m.split(":")[0], int(m.split(':')[1])) for m in line.split(",")[:-1]]
        for line in lines
    ]


def thing(s):
    return [
        [(m.split(":")[0], int(m.split(':')[1])) for m in line.split(",")[:-1]]
        for line in s.splitlines()
    ]


def moves(xs):
    return [move for move, _count in xs]


def picked(xs):
    return max(xs, key=lambda x: x[1])[0]


def sampled(xs):
    return choice(moves(xs), 1, [x[1] for x in xs])


no_exploration = load("no_exploration.txt")[:300]
uniform_beta = load("beta_uniform.txt")[:300]
beta_only = load("beta_only.txt")[:300]
uniform_only = load("uniform_only.txt")[:300]

assert len(no_exploration) == len(uniform_beta) == len(beta_only) == len(uniform_only)

same = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

for no_exp, uni_beta, beta_o, uni_o in zip(
    no_exploration, uniform_only, beta_only, uniform_only
):
    assert len(no_exp) == len(uni_beta) == len(beta_o) == len(uni_o)
    assert moves(no_exp) == moves(uni_beta) == moves(beta_o) == moves(uni_o)

    for _ in range(1000):
        picks = [sampled(no_exp), sampled(uni_beta), sampled(beta_o), sampled(uni_o)]
        for x, a in enumerate(picks):
            for y, b in enumerate(picks):
                if a == b:
                    same[x][y] += 1

print(same)
