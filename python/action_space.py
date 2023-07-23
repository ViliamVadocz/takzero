from memoization import cached


def placements(n):
    if n < 5:
        return 2 * n * n
    return 3 * n * n


def overestimate(n):
    return n * n * (2 ** (n + 1) - 4)


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


if __name__ == "__main__":
    print(
        " n    placements    spreads(over-estimate)    total(over-estimate)    spreads(real)    total(real)"
    )
    for i in range(3, 17):
        p = placements(i)
        o = overestimate(i)
        r = real(i)
        print(f"{i: 3}    {p: 9}    {o: 22}    {p+o: 20}    {r: 13}    {p+r: 11}")
