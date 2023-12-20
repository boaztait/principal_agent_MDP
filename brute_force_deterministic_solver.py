# -*- coding: utf-8 -*-

import numpy

def rec_brute_solve(state, hist, B, VA):
    if len(state.actions) == 0:
        if hist[0] >= VA[0] - B - 0.00001:
            return hist
        return numpy.array([-1, -1])

    best = numpy.array([-1, -1])

    for a in state.actions:
       curr = rec_brute_solve(a.next_state, hist + numpy.array([a.RA, a.RP]), B, VA)
       if curr[1] > best[1]:
           best = curr

    return best

def solve_brute_force(states, B, VA):
    root = states.root
    best = numpy.array([-1, -1])

    for a in root.actions:
        curr = rec_brute_solve(a.next_state, numpy.array([a.RA, a.RP]), B, VA)
        if curr[1] > best[1]:
            best = curr

    return best
