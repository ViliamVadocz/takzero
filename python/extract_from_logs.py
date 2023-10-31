import re
import matplotlib.pyplot as plt

RUN_NAMES = ["uncertainty0", "uncertainty1", "uncertainty2"]
# RUN_NAMES = ["baseline0", "baseline1"]

RND_PATTERN = re.compile(r"rnd=\[(\d*\.\d*)\]")
POLICY_PATTERN = re.compile(r"p=\[(\d*\.\d*)\]")
VALUE_PATTERN = re.compile(r"z=\[(\d*\.\d*)\]")
UBE_PATTERN = re.compile(r"u=\[(\d*\.\d*)\]")

REPLAYS_PATTERN = re.compile(r"Adding (\d*) replays at (\d*) training steps")
TARGETS_PATTERN = re.compile(r"Adding (\d*) targets at (\d*) training steps")


def plot_cumulative(data):
    cumulative = [(0, 0)] + [
        (data[i][1], sum(row[0] for row in data[: i + 1])) for i in range(0, len(data))
    ]
    plt.plot(list(row[0] for row in cumulative), list(row[1] for row in cumulative))


for name in RUN_NAMES:
    with open(f"_data/{name}.err", "r") as file:
        contents = file.read()

    # save data to files
    with open(f"_data/{name}_rnd.txt", "w") as file:
        file.writelines(x[1] + "\n" for x in re.finditer(RND_PATTERN, contents))
    # with open(f"_data/{name}_policy.txt", "w") as file:
    #     file.writelines(x[1] + "\n" for x in re.finditer(POLICY_PATTERN, contents))
    # with open(f"_data/{name}_value.txt", "w") as file:
    #     file.writelines(x[1] + "\n" for x in re.finditer(VALUE_PATTERN, contents))
    # with open(f"_data/{name}_ube.txt", "w") as file:
    #     file.writelines(x[1] + "\n" for x in re.finditer(UBE_PATTERN, contents))

    replays = [(int(x[1]), int(x[2])) for x in re.finditer(REPLAYS_PATTERN, contents)]
    plot_cumulative(replays)
    targets = [(int(x[1]), int(x[2])) for x in re.finditer(TARGETS_PATTERN, contents)]
    plot_cumulative(targets)

plt.xlabel("steps")
plt.ylabel("replays/targets")
plt.savefig("interactions per step")
