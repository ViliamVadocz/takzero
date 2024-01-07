# Put data here after running `grep -c R-0 replays*` etc.
white_road = []
white_flat = []
black_road = []
black_flat = []
draw = []

total = [
    sum(results)
    for results in zip(white_road, white_flat, black_road, black_flat, draw)
]

white_road = [results / total for results, total in zip(white_road, total)]
white_flat = [
    prev + results / total
    for results, total, prev in zip(white_flat, total, white_road)
]
draw = [prev + results / total for results, total, prev in zip(draw, total, white_flat)]
black_flat = [
    prev + results / total for results, total, prev in zip(black_flat, total, draw)
]
black_road = [
    prev + results / total
    for results, total, prev in zip(black_road, total, black_flat)
]

steps = [1000 * x for x in range(len(white_road))]

import matplotlib.pyplot as plt

plt.xlim(steps[0], steps[-1])
plt.ylim(0, 1)

plt.fill_between(steps, black_flat, black_road, color="#000000")
plt.fill_between(steps, draw, black_flat, color="#101010")
plt.fill_between(steps, white_flat, draw, color="#606060")
plt.fill_between(steps, white_road, white_flat, color="#c0c0c0")
plt.fill_between(steps, 0, white_road, color="#e0e0e0")

plt.show()
