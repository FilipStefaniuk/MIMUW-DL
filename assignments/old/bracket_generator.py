import numpy as np
import itertools
import random


def encode(s, max_len):
    encoded = np.zeros((max_len, 2))

    for i, c in enumerate(s):
        if c in ('('):
            encoded[i][0] = 1
        elif c in (')'):
            encoded[i][1] = 1
        else:
            print(c)
            raise ValueError()

    return encoded

def decode(arr):
    return ''.join([')' if x else '(' for x in np.argmax(arr, axis=1)])


def compute_statistics(s):
    max_open, curr_open = 0, 0
    max_consecutive, curr_consecutive = 0, 0
    max_diff, curr_diff = 0, 0

    for c in s:

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


class BracketGenerator:

    def __init__(self, min_len=1, max_len=10, seed=42):
        self.min_len = min_len
        self.max_len = max_len
        random.seed(seed)

    def _random_brackets(self):
        n = random.randint(self.min_len, self.max_len)
        tree = [random.randint(0, x - 1) for x in range(1, n + 1)]
        brackets = [''] * (n + 1)
        for i in reversed(range(len(tree))):
            brackets[tree[i]] += '(' + brackets[i + 1] + ')'
        return brackets[0]

    def __call__(self):
        for _ in itertools.count():
            bracket = self._random_brackets()
            stats = compute_statistics(bracket)
            enc = encode(bracket, self.max_len * 2)
            yield enc, len(bracket), stats
