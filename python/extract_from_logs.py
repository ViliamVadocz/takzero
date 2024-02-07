import re
import os
import matplotlib.pyplot as plt
import numpy as np

POLICY_PATTERN = re.compile(r"loss_policy = \[(\d+\.\d+)\]")
VALUE_PATTERN = re.compile(r"loss_value = \[(\d+\.\d+)\]")
UBE_PATTERN = re.compile(r"loss_ube = \[(\d+\.\d+)\]")
RND_PATTERN = re.compile(r"loss_rnd = \[(\d+\.\d+)\]")

UBE_STATS_REANALYZE_PATTERN = re.compile(
    r"\[UBE STATS\] ply: (\d+), bf: (\d+), root: (\d+\.\d+), max: (\d+\.\d+), target: (\d+\.\d+)"
)
UBE_STATS_SELFPLAY_PATTERN = re.compile(
    r"\[UBE STATS\] ply: (\d+), root: (\d+\.\d+), max: (\d+\.\d+), selected: (\d+\.\d+)"
)

RND_MIN_MAX_PATTERN = re.compile(
    r"Updating RND normalization to min: \[(\d+\.\d+)\] and max: \[(\d+\.\d+)\]"
)


def moving_average(a, n=3):
    l = [x for x in a if x is not None]
    assert len(l) != 0
    ret = np.cumsum(l, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(contents, pattern, title, size=128):
    loss = list(float(x[1]) for x in re.finditer(pattern, contents))
    steps = list(range(len(loss)))

    plt.plot(steps, loss, label="Raw")
    plt.plot(
        steps[(size // 2) : (1 - size // 2)],
        moving_average(loss, size),
        label=f"Moving Average (n={size})",
    )

    plt.title(title)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.ylim(0, loss[-1] * 5)

    plt.legend()
    plt.grid()

    plt.show()


def plot_all_losses(learn_path):
    with open(learn_path, "r") as file:
        contents = file.read()

    plot_loss(contents, POLICY_PATTERN, "Policy Loss During Training")
    plot_loss(contents, VALUE_PATTERN, "Value Loss During Training")
    plot_loss(contents, UBE_PATTERN, "UBE Loss During Training")
    plot_loss(contents, RND_PATTERN, "RND Loss During Training")


def plot_rnd_min_max(learn_path):
    with open(learn_path, "r") as file:
        contents = file.read()

    rnd_min_max = [
        (float(x[1]), float(x[2])) for x in re.finditer(RND_MIN_MAX_PATTERN, contents)
    ]
    print(len(rnd_min_max))

    plt.plot([x[0] for x in rnd_min_max], label="min")
    plt.plot([x[1] for x in rnd_min_max], label="max")

    plt.title("RND reference min and max during training")
    plt.xlabel("Training steps")
    plt.ylabel("MSE")
    plt.ylim(0, rnd_min_max[-1][1] * 2)

    plt.legend()
    plt.grid()
    plt.show()


def get_ube_stats(selfplay: bool, directory: str):
    must_contain = "selfplay" if selfplay else "reanalyze"
    split_text = "Step:" if selfplay else "Number of positions:"
    pattern = UBE_STATS_SELFPLAY_PATTERN if selfplay else UBE_STATS_REANALYZE_PATTERN
    things = 3 if selfplay else 4

    data_per_step = dict()
    for filename in os.listdir(directory):
        if must_contain not in filename:
            continue
        if ".err" not in filename:
            continue
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            print(f"Reading data from {f}")
            with open(f) as file:
                contents = file.read()
            steps = contents.split(split_text)
            for i, step in enumerate(steps[1:]):
                ube_stats = [
                    [int(x[1]), *(float(x[2 + m]) for m in range(things))]
                    for x in re.finditer(pattern, step)
                ]
                data = data_per_step.setdefault(i, [])
                data += ube_stats

    return data_per_step


def plot_ube_vs_bf(directory):
    reanalyze_ube = get_ube_stats(False, directory)

    root_ube_per_bf = dict()
    max_ube_per_bf = dict()
    target_ube_per_bf = dict()
    for data in reanalyze_ube.values():
        for tup in data:
            l = root_ube_per_bf.setdefault(tup[1], [])
            l.append(tup[2])
            l = max_ube_per_bf.setdefault(tup[1], [])
            l.append(tup[3])
            l = target_ube_per_bf.setdefault(tup[1], [])
            l.append(tup[4])
    root_ube_per_bf = sorted(root_ube_per_bf.items())
    max_ube_per_bf = sorted(max_ube_per_bf.items())
    target_ube_per_bf = sorted(target_ube_per_bf.items())

    plt.plot(
        [x for x, _ in root_ube_per_bf],
        [np.mean(y) for _, y in root_ube_per_bf],
        label="root",
    )
    plt.plot(
        [x for x, _ in max_ube_per_bf],
        [np.mean(y) for _, y in max_ube_per_bf],
        label="max",
    )
    plt.plot(
        [x for x, _ in target_ube_per_bf],
        [np.mean(np.sqrt(y)) for _, y in target_ube_per_bf],
        label="target (sqrt)",
    )

    plt.title("Branching factor vs UBE")
    plt.xlabel("Branching factor")
    plt.ylabel("UBE")
    plt.ylim(0, 2)

    plt.legend()
    plt.grid()
    plt.show()


def plot_game_ply_per_bf(directory):
    reanalyze_ube = get_ube_stats(False, directory)

    bf_per_ply = dict()
    for data in reanalyze_ube.values():
        for tup in data:
            l = bf_per_ply.setdefault(tup[0], [])
            l.append(tup[1])
    bf_per_ply = sorted(bf_per_ply.items())

    plt.plot(
        [x for x, y in bf_per_ply][::2],
        [np.mean(y) for x, y in bf_per_ply][::2],
        label="white",
    )
    plt.plot(
        [x for x, y in bf_per_ply][1::2],
        [np.mean(y) for x, y in bf_per_ply][1::2],
        label="black",
    )

    plt.title("Game ply vs branching factor")
    plt.xlabel("Game ply")
    plt.ylabel("Branching factor")

    plt.legend()
    plt.grid()
    plt.show()


def plot_ube_over_plies_across_training(
    number_of_training_ranges: int, selfplay: bool, directory: str
):
    ube_stats = list(get_ube_stats(selfplay, directory).values())
    root_ube_index = 1 if selfplay else 2
    title = "Selfplay" if selfplay else "Reanalyze"

    step_size = len(ube_stats) / number_of_training_ranges
    for i in range(number_of_training_ranges):
        start = int(step_size * i)
        end = int(step_size * (i + 1))
        ube_per_ply = {i: [] for i in range(150)}
        for data in ube_stats[start:end]:
            for tup in data:
                ube_per_ply[tup[0]].append(tup[root_ube_index])
        root = [
            (i, np.mean(ube), np.std(ube))
            for i, ube in ube_per_ply.items()
            if len(ube) > 0
        ]

        if number_of_training_ranges == 1:
            white = root[::2]
            black = root[1::2]
            plt.plot([x for x, _, _ in white], [y for _, y, _ in white], label=f"white")
            plt.fill_between(
                [x for x, _, _ in white],
                [y - z for _, y, z in white],
                [y + z for _, y, z in white],
                alpha=0.2,
            )
            plt.plot([x for x, _, _ in black], [y for _, y, _ in black], label=f"black")
            plt.fill_between(
                [x for x, _, _ in black],
                [y - z for _, y, z in black],
                [y + z for _, y, z in black],
                alpha=0.2,
            )
        else:
            plt.plot(
                [x for x, _, _ in root],
                [y for _, y, _ in root],
                label=f"[{start},{end})",
            )

    if number_of_training_ranges == 1:
        plt.title(f"Root UBE During {title}, Over Game Plies, Split Between Color")
    else:
        plt.title(
            f"Root UBE During {title}, Over Game Plies, For Several Training Periods"
        )
    plt.xlabel("Game plies (half-moves)")
    plt.ylabel("Average UBE")
    plt.ylim(0)

    plt.legend()
    plt.grid()
    plt.show()


def mean_and_std(data, ply_step, i):
    l = [tup[2] for tup in data if i * ply_step <= tup[0] < (i + 1) * ply_step]
    if len(l) == 0:
        return None, None
    return (np.mean(l), np.std(l))


def plot_ube_over_training_across_plies(
    ply_step, num_steps, selfplay, directory, size=128
):
    ube_stats = list(get_ube_stats(selfplay, directory).values())
    title = "Selfplay" if selfplay else "Reanalyze"

    for i in range(num_steps):
        root = [mean_and_std(data, ply_step, i) for data in ube_stats]
        ube = moving_average([a for a, _ in root], size)
        std = moving_average([b for _, b in root], size)
        plt.plot(
            ube,
            label=f"[{i * ply_step},{(i + 1) * ply_step})",
        )
        if num_steps == 1:
            plt.fill_between(
                list(range(len(ube))),
                [a - b for a, b in zip(ube, std)],
                [a + b for a, b in zip(ube, std)],
                alpha=0.2,
            )

    plt.title(f"Root UBE During {title}, for several game ply (half-move) ranges")
    plt.xlabel("Steps")
    plt.ylabel(f"Root UBE (Moving Average size={size})")
    plt.ylim(0, 2.5)

    plt.legend()
    plt.grid()
    plt.show()


def length_of_game_from_line(line: str):
    split = line.split("]")
    if len(split) < 2:
        return None
    return len(line.split("]")[1].strip().split()) + 1


def get_game_lengths(replay_path):
    with open(replay_path, "r") as file:
        return [
            length_of_game_from_line(line)
            for line in file.readlines()
            if len(line.split("]")) > 1
        ]


def plot_all_game_lengths(replay_path):
    plt.hist(get_game_lengths(replay_path), bins=[x for x in range(130)], density=True)
    plt.title("Histogram of Game Length")
    plt.xlabel("Game Plies (Half-Moves)")
    plt.ylabel("Density")
    plt.show()


def plot_all_game_lengths_per_color(replay_path, max_len=130):
    game_lengths = get_game_lengths(replay_path)
    plt.hist(
        [l for l in game_lengths if l % 2 == 1],
        bins=[x for x in range(max_len)],
        density=True,
        label="white",
        # alpha=0.5,
    )
    plt.hist(
        [l for l in game_lengths if l % 2 == 0],
        bins=[x for x in range(max_len)],
        density=True,
        label="black",
        # alpha=0.5,
    )
    plt.title("Histogram of Game Length")
    plt.xlabel("Game Plies (Half-Moves)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


from enum import Enum


class WinType(Enum):
    WHITE_ROAD = 1
    WHITE_FLAT = 2
    DRAW = 3
    BLACK_FLAT = 4
    BLACK_ROAD = 5

    def from_line(line):
        if "R-0" in line:
            return WinType.WHITE_ROAD
        if "F-0" in line:
            return WinType.WHITE_FLAT
        if "1/2-1/2" in line:
            return WinType.DRAW
        if "0-F" in line:
            return WinType.BLACK_FLAT
        if "0-R" in line:
            return WinType.BLACK_ROAD
        return None


def get_wins(replay_path):
    with open(replay_path, "r") as file:
        return [
            x
            for x in (WinType.from_line(line) for line in file.readlines())
            if x is not None
        ]


def get_wins_within_range(replay_path, min_length, max_length):
    with open(replay_path, "r") as file:
        return [
            x
            for x, l in (
                (WinType.from_line(line), length_of_game_from_line(line))
                for line in file.readlines()
            )
            if x is not None and l is not None and min_length <= l < max_length
        ]


def chunks(seq, n):
    return (seq[i : i + n] for i in range(0, len(seq), n))


def win_rate(wins, period, win_type):
    return [
        sum(1 for w in chunk if w == win_type) / period
        for chunk in chunks(wins, period)
    ]


def plot_win_rate(wins, period):
    wins = wins[: len(wins) - len(wins) % period]

    black_road = win_rate(wins, period, WinType.BLACK_ROAD)
    black_flat = win_rate(wins, period, WinType.BLACK_FLAT)
    draw = win_rate(wins, period, WinType.DRAW)
    white_flat = win_rate(wins, period, WinType.WHITE_FLAT)
    white_road = win_rate(wins, period, WinType.WHITE_ROAD)
    steps = range(0, len(wins), period)

    white_flat = [prev + results for results, prev in zip(white_flat, white_road)]
    draw = [prev + results for results, prev in zip(draw, white_flat)]
    black_flat = [prev + results for results, prev in zip(black_flat, draw)]
    black_road = [prev + results for results, prev in zip(black_road, black_flat)]

    plt.title("Win-rates and Win-types During Training")
    plt.xlabel("Replay Number")
    plt.ylabel("Win-rate")
    plt.xlim(steps[0], steps[-1])
    plt.ylim(0, 1)

    plt.fill_between(steps, black_flat, black_road, color="#000000", label="black road")
    plt.fill_between(steps, draw, black_flat, color="#202020", label="black flat")
    plt.fill_between(steps, white_flat, draw, color="#606060", label="draw")
    plt.fill_between(steps, white_road, white_flat, color="#c0c0c0", label="white flat")
    plt.fill_between(steps, 0, white_road, color="#e0e0e0", label="white road")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_all_losses("out/learn-9277117.err")
    # plot_rnd_min_max("out/learn-9277117.err")
    # plot_ube_over_plies_across_training(1, True, "./out")
    # plot_ube_over_training_across_plies(100, 1, True, "./out")
    # plot_game_ply_per_bf("./out")
    # plot_all_game_lengths("./replays.txt")
    # plot_all_game_lengths_per_color("./replays.txt", max_len=80)
    # plot_ube_vs_bf("./out")
    # plot_win_rate(get_wins("replays.txt"), 1000)
    # wins = get_wins_within_range("replays.txt", 0, 30)
    # plot_win_rate(wins, 1000)
    # wins = get_wins_within_range("replays.txt", 30, 120)
    # plot_win_rate(wins, 1000)
    pass
