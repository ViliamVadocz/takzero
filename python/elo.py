from dataclasses import dataclass
from subprocess import Popen, PIPE


@dataclass
class MatchResult:
    a: int
    b: int
    wins: int
    losses: int
    draws: int

    @staticmethod
    def from_line(line: str) -> "MatchResult":
        [a, b, wins, losses, draws] = line.split(", ")
        return MatchResult(int(a), int(b), int(wins), int(losses), int(draws))


with open("match_results_exploration.csv") as f:
    exploration = [MatchResult.from_line(line) for line in f.readlines()]
    exploration_players = {a: i for (i, a) in enumerate({m.a for m in exploration})}
with open("match_results_baseline.csv") as f:
    baseline = [MatchResult.from_line(line) for line in f.readlines()]
    l = len(exploration_players)
    baseline_players = {a: i + l for (i, a) in enumerate({m.a for m in baseline})}

proc = Popen(
    [".\\python\\bayeselo.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
)
proc.stdin.write("prompt off\n")

for p in exploration_players.keys():
    proc.stdin.write(f"addplayer exploration_{p}\n")
for p in baseline_players.keys():
    proc.stdin.write(f"addplayer baseline_{p}\n")

print("added players")

for m in exploration:
    white = exploration_players[m.a]
    black = exploration_players[m.b]
    for _ in range(m.wins):
        proc.stdin.write(f"addresult {white} {black} 2\n")
    for _ in range(m.losses):
        proc.stdin.write(f"addresult {white} {black} 0\n")
    for _ in range(m.draws):
        proc.stdin.write(f"addresult {white} {black} 1\n")

print("added exploration matches")

for m in baseline:
    white = baseline_players[m.a]
    black = baseline_players[m.b]
    for _ in range(m.wins):
        proc.stdin.write(f"addresult {white} {black} 2\n")
    for _ in range(m.losses):
        proc.stdin.write(f"addresult {white} {black} 0\n")
    for _ in range(m.draws):
        proc.stdin.write(f"addresult {white} {black} 1\n")

print("added baseline matches")

comparison = [
    (0, 35, 24, 5),
    (50000, 20, 29, 15),
    (100000, 25, 30, 9),
    (150000, 29, 18, 17),
    (200000, 22, 23, 19),
]

for i, b_wins, e_wins, draws in comparison:
    proc.stdin.flush()
    exploration = exploration_players[i]
    baseline = baseline_players[i]
    for i in range(b_wins):
        if i % 2 == 0:
            proc.stdin.write(f"addresult {baseline} {exploration} 2\n")
        else:
            proc.stdin.write(f"addresult {exploration} {baseline} 0\n")
    for i in range(e_wins):
        if i % 2 == 0:
            proc.stdin.write(f"addresult {exploration} {baseline} 2\n")
        else:
            proc.stdin.write(f"addresult {baseline} {exploration} 0\n")
    for i in range(draws):
        if i % 2 == 0:
            proc.stdin.write(f"addresult {baseline} {exploration} 1\n")
        else:
            proc.stdin.write(f"addresult {exploration} {baseline} 1\n")

print("added comparison matches")

# proc.stdin.write("players\n")
# proc.stdin.write("results\n")

proc.stdin.write("elo\n")  # enter elo interface
proc.stdin.write("mm\n")  # compute maximum-likelyhood elo
proc.stdin.write("ratings\n")  # print out ratings
proc.stdin.write("x\n")  # leave elo interface
out = proc.communicate(input="x\n")[0]  # close application

import re

elo = [
    (m[1].split("_"), int(m[2]))
    for m in re.finditer(re.compile(r"(\w+_\d+)\s+(-?\d+)"), out)
]

baseline = sorted([(int(digits), e) for (name, digits), e in elo if "baseline" in name])
exploration = sorted(
    [(int(digits), e) for (name, digits), e in elo if "exploration" in name]
)

import matplotlib.pyplot as plt

plt.plot([x[0] for x in baseline], [x[1] for x in baseline], label="baseline")
plt.plot([x[0] for x in exploration], [x[1] for x in exploration], label="exploration")
plt.grid()
plt.legend()
plt.xlabel("training steps")
plt.ylabel("relative bayes elo")
plt.show()
