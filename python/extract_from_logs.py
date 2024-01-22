import re
import matplotlib.pyplot as plt
import numpy as np

POLICY_PATTERN = re.compile(r"loss_policy = \[(\d+\.\d+)\]")
VALUE_PATTERN = re.compile(r"loss_value = \[(\d+\.\d+)\]")
UBE_PATTERN = re.compile(r"loss_ube = \[(\d+\.\d+)\]")
RND_PATTERN = re.compile(r"loss_rnd = \[(\d+\.\d+)\]")

UBE_STATS_PATTERN = re.compile(
    r"\[UBE STATS\] root: (\d+\.\d+), max: (\d+\.\d+), selected: (\d+\.\d+)"
)
BATCH_SIZE = 128


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# with open("_data/learn-9230002.err", "r") as file:
#     contents = file.read()
# policy_loss = list(float(x[1]) for x in re.finditer(POLICY_PATTERN, contents))
# value_loss = list(float(x[1]) for x in re.finditer(VALUE_PATTERN, contents))
# ube_loss = list(float(x[1]) for x in re.finditer(UBE_PATTERN, contents))
# rnd_loss = list(float(x[1]) for x in re.finditer(RND_PATTERN, contents))

# num_steps = len(ube_loss)
# steps = list(range(num_steps))

# plt.title("UBE loss")
# plt.xlabel("training steps")
# plt.ylabel("loss")
# plt.plot(steps, ube_loss)
# plt.ylim(0, 0.4)

with open("_data/selfplay-9230004.err", "r") as file:
    contents = file.read()
ube_stats = [
    (float(x[1]), float(x[2]), float(x[3]))
    for x in re.finditer(UBE_STATS_PATTERN, contents)
]
root_ube = [t[0] for t in ube_stats]
max_ube = [t[1] for t in ube_stats]
selected_ube = [t[2] for t in ube_stats]

plt.plot(moving_average(root_ube, n=2048), label="root")
plt.plot(moving_average(max_ube, n=2048), label="max")
plt.plot(moving_average(selected_ube, n=2048), label="selected")
plt.ylabel("moving average UBE")
plt.xlabel("self-play steps")
plt.legend()

plt.show()
