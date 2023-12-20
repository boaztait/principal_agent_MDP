# -*- coding: utf-8 -*-
import numpy

# =============================================================================
# Find the vector in the pareto frontier that corresponds to the maximum VP
# =============================================================================
def find_opt_VP(U, B, VA):
    max_u = [0, 0]

    for u in U:
        if u[1] > max_u[1]:
            if u[0] >= VA[0] - B:
                max_u = u

    return max_u

def sortkey(ele):
    return ele[0]

def maxkey(ele):
    return ele[1]

# =============================================================================
# Calculate the pareto frontier from set U
# =============================================================================
def pareto_front(U):
    if len(U) == 0:
        return U

    new_U = []
    U.sort(key=sortkey, reverse=True)
    U.append(numpy.array([0, 0]))

    start_idx = 0
    global_max = -1
    curr = U[0][0];
    for i in range(len(U)):
        if curr != U[i][0]:
            curr_max = max(U[start_idx:i], key=maxkey)
            if curr_max[1] > global_max:
                new_U.append(curr_max)
                global_max = curr_max[1]

            start_idx = i;
            curr = U[i][0]

    return new_U

# =============================================================================
# descretize vector u according to eps
# =============================================================================
def desc(u, eps):
    return numpy.floor(u / eps) * eps

# =============================================================================
# Recursively perform the loop of algorithm DFAR
# =============================================================================
def do_inner_DFAR(states, eps):
    U = []

    for a in states.actions:
        a_utility = desc(numpy.array([a.RA, a.RP]), eps)
        if len(a.next_state.U) == 0:
            U.append(a_utility)
        for utility in a.next_state.U:
            U.append(a_utility + utility)

    states.U = pareto_front(U)

def contains_array(arr, u):
    for uarr in u:
        if uarr[0] == arr[0] and uarr[1] == arr[1]:
            return 1
    return 0

def find_dfar_u_path(ddp_states, layers, eps, dfar_u):
    a_list = []

    state = ddp_states.root
    next_u = dfar_u

    while len(a_list) < layers:
        to_loop = 1
        for a in state.actions:
            a_utility = desc(numpy.array([a.RA, a.RP]), eps)

            if a_utility[0] == next_u[0] and a_utility[1] == next_u[1]:
                a_list.append(a)
                break

            for u in a.next_state.U:
                u_vec = u + a_utility
                if u_vec[0] == next_u[0] and u_vec[1] == next_u[1]:
                    a_list.append(a)
                    state = a.next_state
                    next_u = u
                    to_loop = 0
                    break

            if to_loop == 0:
                break


    return a_list


def DFAR(ddp_states, layers, nodes_in_layer, B, eps, VA):
    for l in range(layers):
        for k in range(nodes_in_layer):
            do_inner_DFAR(ddp_states.states[l][k], eps)
    do_inner_DFAR(ddp_states.root, eps)

    opt_dfar = find_opt_VP(ddp_states.root.U, B, VA)

    if opt_dfar[0] == 0 and opt_dfar[1] == 0:
        return VA[1]

    a_list = find_dfar_u_path(ddp_states, layers, eps, opt_dfar)

    opt_vp = 0
    for a in a_list:
        opt_vp = opt_vp + a.RP

    return opt_vp

