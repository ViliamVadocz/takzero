from dataclasses import dataclass
from subprocess import Popen, PIPE


@dataclass
class MatchResult:
    a: str
    b: str
    wins: int
    losses: int
    draws: int

    @staticmethod
    def from_line(line: str, prefix: str) -> "MatchResult":
        [a, b, wins, losses, draws] = line.split(", ")
        return MatchResult(
            f"{prefix}_{int(a)}",
            f"{prefix}_{int(b)}",
            int(wins),
            int(losses),
            int(draws),
        )


def read_results(path: str, players: set[str], prefix: str) -> list[MatchResult]:
    with open(path) as f:
        matches = [MatchResult.from_line(line, prefix) for line in f.readlines()]
        players |= {m.a for m in matches} | {m.b for m in matches}
    return matches


def add_players(proc, players: set[str]) -> dict[str, int]:
    new_players = dict()
    for i, p in enumerate(players):
        proc.stdin.write(f"addplayer {p}\n")
        new_players[p] = i
    return new_players


def add_matches(proc, matches: list[MatchResult], players: dict[str, int]):
    for m in matches:
        add_match(proc, m, players)


def add_match(proc, match: MatchResult, players: dict[str, str]):
    white = players[match.a]
    black = players[match.b]
    for _ in range(match.wins):
        proc.stdin.write(f"addresult {white} {black} 2\n")
    for _ in range(match.losses):
        proc.stdin.write(f"addresult {white} {black} 0\n")
    for _ in range(match.draws):
        proc.stdin.write(f"addresult {white} {black} 1\n")


proc = Popen(
    [".\\python\\bayeselo.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
)
proc.stdin.write("prompt off\n")

players = set()
exploration = read_results("match_results_exploration.csv", players, "exploration")
baseline = read_results("match_results_baseline.csv", players, "baseline")
no_explore = read_results("match_results_no_explore.csv", players, "no_explore")
print("read results")

players = add_players(
    proc,
    players,
    # {p for p in players if (x := int(p.split("_")[-1])) % 50_000 == 0 and x <= 200_000},
)
print("added players")

# proc.stdin.write("players\n")
# proc.stdin.write("results\n")
# exit()

add_matches(proc, exploration, players)
add_matches(proc, baseline, players)
add_matches(proc, no_explore, players)
print("added intra-agent matches")

with open("match_results_tournament.csv") as f:
    for a, b, wins, losses, draws in (line.split(", ") for line in f.readlines()):
        add_match(proc, MatchResult(a, b, int(wins), int(losses), int(draws)), players)
print("added inter-agent matches")

# proc.stdin.write("players\n")
# proc.stdin.write("results\n")
# exit()

proc.stdin.write("elo\n")  # enter elo interface
proc.stdin.write("mm\n")  # compute maximum-likelyhood elo
proc.stdin.write("ratings\n")  # print out ratings
proc.stdin.write("x\n")  # leave elo interface
out = proc.communicate(input="x\n")[0]  # close application

print(out)

import re

elo = [
    (m[1], int(m[2])) for m in re.finditer(re.compile(r"([\w_]+\d+)\s+(-?\d+)"), out)
]

baseline = sorted(
    [(int(name.split("_")[1]), e) for name, e in elo if "baseline" in name]
)
exploration = sorted(
    [(int(name.split("_")[1]), e) for name, e in elo if "exploration" in name]
)
no_explore = sorted(
    [(int(name.split("_")[2]), e) for name, e in elo if "no_explore" in name]
)

import matplotlib.pyplot as plt

plt.plot([x[0] for x in baseline], [x[1] for x in baseline], label="baseline")
plt.plot([x[0] for x in exploration], [x[1] for x in exploration], label="exploration")
plt.plot([x[0] for x in no_explore], [x[1] for x in no_explore], label="no_explore")
plt.grid()
plt.legend()
plt.xlabel("training steps")
plt.ylabel("relative bayes elo")
plt.show()
