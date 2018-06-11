import numpy as np


def random_bracket(n):
    tree = [np.random.randint(low=0, high=x + 1) for x in range(n)]
    brackets = [''] * (n + 1)
    for i in reversed(range(len(tree))):
        brackets[tree[i]] += '(' + brackets[i + 1] + ')'
    return brackets[0]


def random_bracket_heuristic(n, p=0.6, q=0.8):

    tree = [0, 0]
    curr = 1
    depth = 0

    for i in range(2, n + 1):
        r = np.random.rand()

        if r < p * (depth / n) * q:
            tree.append(tree[tree[curr]])
            depth = max(depth - 1, 0)

        elif r < p:
            tree.append(curr)
            depth += 1
        else:
            tree.append(tree[curr])

        curr = i

    tree = tree[1:]

    brackets = [''] * (n + 1)
    for i in reversed(range(len(tree))):
        brackets[tree[i]] += '(' + brackets[i + 1] + ')'
    return brackets[0]


def random_bracket_joined(n, h=0.5, **kwargs):
    r = np.random.rand()
    if r > h:
        return random_bracket_heuristic(n, **kwargs)
    return random_bracket(n)


def random_bracket_len(n, min_len=1):
    n = n - ((np.random.zipf(1.4) - 1) % (n - min_len))
    return random_bracket_joined(n)


def statistics(brackets):
    max_open, curr_open = 0, 0
    max_consecutive, curr_consecutive = 0, 0
    max_diff, curr_diff = 0, 0

    for c in brackets:
        if c == '(':
            curr_open += 1
            max_open = max(max_open, curr_open)

            curr_consecutive += 1
            max_consecutive = max(max_consecutive, curr_consecutive)

            curr_diff += 1

        elif c == ')':
            curr_open -= 1
            curr_consecutive = 0

            if curr_open == 0:
                max_diff = max(max_diff, curr_diff)
                curr_diff = 0
            else:
                curr_diff += 1

    return max_open, max_consecutive, max_diff


def next_batch(size, n):
    bs = [random_bracket_len(n) for _ in range(size)]
    stats = [statistics(b) for b in bs]
    return np.array(bs), np.array(stats)


def parse_string(s):
    s = s.strip()
    s = ''.join(s.split())

    count = 0
    for i in s:
        if i == "(":
            count += 1
        elif i == ")":
            count -= 1
        else:
            return False

    if count == 0:
        return s
    else:
        raise ValueError("bad input")
