from dataclasses import dataclass
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import re

# Result format:
# <white>, <white_steps>, <black>, <black_steps>, <wins>, <losses>, <draws>


@dataclass
class MatchResult:
    white: str
    white_steps: int
    black: str
    black_steps: int
    wins: int
    losses: int
    draws: int

    @staticmethod
    def from_line(line: str) -> "MatchResult":
        [white, white_steps, black, black_steps, wins, losses, draws] = line.split(", ")
        return MatchResult(
            white,
            int(white_steps),
            black,
            int(black_steps),
            int(wins),
            int(losses),
            int(draws),
        )

    def white_name(self) -> str:
        return name(self.white, self.white_steps)

    def black_name(self) -> str:
        return name(self.black, self.black_steps)


def name(model: str, steps: int) -> str:
    return f"{model}_{steps}"


def read_results(*paths: str) -> list[MatchResult]:
    results: list[MatchResult] = []
    for path in paths:
        with open(path) as f:
            results.extend(MatchResult.from_line(line) for line in f.readlines())
    return results


def add_players(proc, player_set: set[str]) -> dict[str, int]:
    players = dict()
    for i, player in enumerate(player_set):
        proc.stdin.write(f"addplayer {player}\n")
        players[player] = i
    return players


def add_matches(proc, matches: list[MatchResult], players: dict[str, int]):
    for m in matches:
        add_match(proc, m, players)


def add_match(proc, match: MatchResult, players: dict[str, int]):
    white = players[match.white_name()]
    black = players[match.black_name()]
    for _ in range(match.wins):
        proc.stdin.write(f"addresult {white} {black} 2\n")
    for _ in range(match.losses):
        proc.stdin.write(f"addresult {white} {black} 0\n")
    for _ in range(match.draws):
        proc.stdin.write(f"addresult {white} {black} 1\n")


proc = Popen(
    [".\\python\\bayeselo.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
)
assert proc.stdin is not None
proc.stdin.write("prompt off\n")

match_results = read_results(
    # "match_results_exploration.csv",
    # "match_results_baseline.csv",
    # "match_results_no_explore.csv",
    # "match_results_big_baseline.csv",
    "match_results.csv",
    "match_results_tournament.csv",
)
print("read results")

player_set = {m.white_name() for m in match_results}
player_set |= {m.black_name() for m in match_results}
players = add_players(proc, player_set)
print("added players")

add_matches(proc, match_results, players)
print("added matches")

# proc.stdin.write("players\n")
# proc.stdin.write("results\n")
# exit()

proc.stdin.write("elo\n")  # enter elo interface
proc.stdin.write("mm\n")  # compute maximum-likelyhood elo
proc.stdin.write("ratings\n")  # print out ratings
proc.stdin.write("x\n")  # leave elo interface
out = proc.communicate(input="x\n")[0]  # close application

print(out)

elo = {m[1]: int(m[2]) for m in re.finditer(re.compile(r"([\w_-]+\d+)\s+(-?\d+)"), out)}

models = {m.white for m in match_results} | {m.black for m in match_results}
model_steps = {
    model: {m.white_steps for m in match_results if m.white == model}
    | {m.black_steps for m in match_results if m.black == model}
    for model in models
}

for model, steps in model_steps.items():
    steps_sorted = sorted(steps)
    model_elo = [elo[name(model, step)] for step in steps_sorted]
    plt.plot(steps_sorted, model_elo, label=model)

plt.legend()
plt.xlabel("training steps")
plt.ylabel("relative bayes elo")
plt.show()
