from memoization import cached


def placements(n):
    if n < 5:
        return 2 * n * n
    return 3 * n * n


def overestimate(n):
    return n * n * 4 * (2**n - 1)


def real(n):
    return sum(
        sum(f(n, x) + f(n, y) + f(n, n - x - 1) + f(n, n - y - 1) for y in range(0, n))
        for x in range(0, n)
    )


@cached
def f(n, k):
    return spread_n_into_k(n, k)


def spread_n_into_k(n, k):
    return sum(spread_exactly_n_into_k(i, k) for i in range(1, n + 1))


def spread_exactly_n_into_k(n, k):
    return sum(spread_exactly_n_into_exactly_k(n, i) for i in range(1, k + 1))


@cached
def spread_exactly_n_into_exactly_k(n, k):
    if n < k:
        return 0
    if k == n or k == 1:
        return 1
    return sum(
        spread_exactly_n_into_exactly_k(n - i, k - 1) for i in range(1, n - k + 2)
    )


@cached
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def n_choose_k(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)


def stars_and_bars(flats, squares):
    return n_choose_k(flats + squares - 1, flats)


if __name__ == "__main__":
    from math import log10

    flat_count = [0, 0, 0, 10, 15, 21, 30, 40, 50]
    cap_count = [0, 0, 0, 0, 0, 1, 1, 2, 2]

    print("n  upper")
    for n in range(3, 9):
        # under = (
        #     (n * n) ** (2 * flat_count[n] - 1)
        #     / factorial(flat_count[n])
        #     / factorial(flat_count[n] - 1)
        # )
        fc = flat_count[n]
        cc = cap_count[n]

        lower = sum(
            (n * n) ** (white_count + black_count)
            / factorial(white_count)
            / factorial(black_count)
            for white_count in range(fc - 1)
            for black_count in range(fc - 1)
        )
        upper = (
            # division points
            (factorial(2 * fc + 1 + n * n) // factorial(2 * fc + 1) // factorial(n * n))
            # number of ways to arrange the stack
            * (factorial(2 * fc) // factorial(fc) // factorial(fc))
            # walls or not walls
            * 2 ** (n * n)
            # capstones
            * (n * n + 1) ** (2 * cc)
            # color to move
            * 2
        )

        squares = n * n
        upper_better = (
            1
            + squares
            + 2
            * sum(
                # division into stacks on squares
                stars_and_bars(white_flats + black_flats, squares)
                # flats of same color are identical
                * factorial(white_flats + black_flats)
                // factorial(white_flats)
                // factorial(black_flats)
                # placing caps and walls
                * n_choose_k(
                    squares, white_caps + black_caps + white_walls + black_walls
                )
                * factorial(white_caps + black_caps + white_walls + black_walls)
                # caps of same color are identical
                // factorial(white_caps) // factorial(black_caps)
                # walls of same color are identical
                // factorial(white_walls) // factorial(black_walls)
                # color to move
                for white_used in range(1, fc + 1)
                for black_used in range(1, fc + 1 if white_used != fc else fc)
                for white_caps in range(cc + 1)
                for black_caps in range(cc + 1)
                for white_walls in range(min(white_used - 1, squares - white_caps) + 1)
                for black_walls in range(min(black_used - 1, squares - black_caps) + 1)
                for white_flats in [white_used - white_walls]
                for black_flats in [black_used - black_walls]
            )
        )
        l10 = int(log10(upper_better))
        s = upper_better / 10**l10
        print(f"{n}  {s:.6} × 10^{int(log10(upper_better)):3}   {upper_better}")

    # print(
    #     " n    placements    spreads(over-estimate)    total(over-estimate)    spreads(real)    total(real)"
    # )
    # for i in range(3, 17):
    #     p = placements(i)
    #     o = overestimate(i)
    #     r = real(i)
    #     print(f"{i: 3}    {p: 9}    {o: 22}    {p+o: 20}    {r: 13}    {p+r: 11}")
