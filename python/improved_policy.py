from dataclasses import dataclass
from math import sqrt
from scipy.special import softmax

BETA = 0.0


def to_value(s: str):
    if "Win" in s:
        return 1
    if "Loss" in s:
        return -1
    if "Draw" in s:
        return 0
    return float(s)


@dataclass
class Action:
    string: str
    visits: int
    value: float
    std_dev: float
    logit: float

    def q(self):
        return self.value + BETA * self.std_dev

    def from_string(s):
        [string, visits, eval, std_dev, logit] = s.split(":")
        return Action(
            string, int(visits), -to_value(eval), float(std_dev), float(logit)
        )

    def sigma1(self, visits):
        return self.q() * (50 + visits)

    def sigma2(self, visits):
        return self.q() * visits

    def sigma3(self, visits):
        return self.q() * sqrt(visits)

    def sigma4(self, visits):
        return self.q() * sqrt(visits) * 0.5


def load(path):
    with open(path) as file:
        lines = file.readlines()
    return [[Action.from_string(m) for m in line.split(",")[:-1]] for line in lines]


run = load("seqhal_64_no_beta_linear_50_puct.txt")

for xs in run:
    xs.sort(key=lambda x: x.visits, reverse=True)
    max_visits = xs[0].visits
    i1 = softmax([x.logit + x.sigma1(max_visits) for x in xs])
    i2 = softmax([x.logit + x.sigma2(max_visits) for x in xs])
    i3 = softmax([x.logit + x.sigma3(max_visits) for x in xs])
    i4 = softmax([x.logit + x.sigma4(max_visits) for x in xs])

    data = [
        [x.visits for x in xs],
        [x.logit for x in xs],
        [x.q() for x in xs],
        i1,
        i2,
        i3,
        i4,
    ]

    print("visits,logit,q+beta*std_dev,i1,i2,i3,i4")
    for i in range(len(data[0])):
        print(",".join(str(d[i]) for d in data))

    input("Press enter.")
