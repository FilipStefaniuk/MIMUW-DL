# from bracket_generator import BracketGenerator
# import sys

import brackets

if __name__ == '__main__':
    # gen = BracketGenerator(max_len=10)
    # bs = {}
    # steps = 0

    # for b, _ in gen():
    #     bs[b] = bs.get(b, 0) + 1

    #     if steps == 10000:
    #         break
    #     else:
    #         steps += 1

    # print(brackets.random_bracket_2(3, 3))
    print(brackets.random_bracket_len(10))
