#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class MinDSU:
    def __init__(self, element_count):
        self.parent = [i for i in range(element_count)]
        self.value = [i for i in range(element_count)]
        self.size = [1 for i in range(element_count)]
    
    def get_representative(self, element):
        path = [element]
        while self.parent[element] != element:
            element = self.parent[element]
            path.append(element)
        for descendant in path:
            self.parent[descendant] = element
        return element
    
    def union(self, a, b):
        a_parent = self.get_representative(a)
        b_parent = self.get_representative(b)
        if a_parent == b_parent:
            return
        if self.size[a_parent] < self.size[b_parent]:
            a_parent, b_parent = b_parent, a_parent
        self.parent[b_parent] = a_parent
        self.size[a_parent] += self.size[b_parent]
        self.value[a_parent] = min(self.value[a_parent], self.value[b_parent])
    
    def get_min(self, element):
        return self.value[self.get_representative(element)]
 

def calc_schedule(deadlines):
    n = len(deadlines)
    schedule = [None for i in range(n)]
    occupied_blocks = MinDSU(n)
    for task_num in range(n - 1, -1, -1):
        d = deadlines[task_num]
        if schedule[d] is None:
            place = d
        else:
            place = occupied_blocks.get_min(d) - 1
            if place < 0:
                return None
        assert schedule[place] is None
        schedule[place] = task_num
        if place > 0 and schedule[place - 1] is not None:
            occupied_blocks.union(place, place - 1)
        if place + 1 < n and schedule[place + 1] is not None:
            occupied_blocks.union(place, place + 1)
    assert all(task is not None for task in schedule)
    return schedule
