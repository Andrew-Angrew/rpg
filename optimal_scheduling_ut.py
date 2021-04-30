#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import permutations
from random import randint, seed

from optimal_scheduling import MinDSU, calc_schedule

def test_dsu():
    dsu = MinDSU(7)
    for elem in range(7):
        assert dsu.get_min(elem) == elem
    dsu.union(2, 5)
    dsu.union(3, 5)
    assert dsu.get_min(3) == 2
    assert dsu.get_min(6) == 6
    assert dsu.get_min(5) == 2

def test_scheduling_simple():
    deadlines = [5, 5, 5, 2, 2, 2]
    assert calc_schedule(deadlines) == [3, 4, 5, 0, 1, 2]

    deadlines = [5, 5, 4, 2, 2, 2]
    assert calc_schedule(deadlines) == [3, 4, 5, 0, 2, 1]


def calc_movement(schedule):
    return sum(abs(i - task_num) for i, task_num in enumerate(schedule))

def calc_optimal_movement(deadlines):
    n = len(deadlines)
    assert n < 10
    min_movement = None
    for schedule in permutations(range(n)):
        if any(deadlines[task_num] < pos for pos, task_num in enumerate(schedule)):
            continue
        movement = calc_movement(schedule)
        if min_movement is None or min_movement > movement:
            min_movement = movement
    return min_movement

def test_scheduling():
    seed(0)
    n = 7
    for test_id in range(100):
        deadlines = [randint(1, n - 1) for i in range(n)]
        opt = calc_optimal_movement(deadlines)
        schedule = calc_schedule(deadlines)
        if opt is None:
            assert schedule is None
        else:
            assert schedule is not None
            assert calc_movement(schedule) == opt, (calc_movement(schedule), opt)

if __name__ == "__main__":
    test_dsu()
    test_scheduling_simple()
    test_scheduling()