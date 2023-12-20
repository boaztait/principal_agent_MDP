# -*- coding: utf-8 -*-

import random
import numpy
import math
import pandas as pd

import up_mdp_deterministic_solver as u_solver
import brute_force_deterministic_solver as bf_solver

def maxVA(ele):
    return ele[0]

def get_VA(ddp_states, layers, nodes_in_layer):
    for l in range(layers):
        for k in range(nodes_in_layer):
            state = ddp_states.states[l][k]
            temp = [numpy.array([0, 0])]

            for a in state.actions:
                temp.append(numpy.array([a.RA, a.RP]) + a.next_state.VA)

            state.VA = max(temp, key=maxVA)

    temp = [numpy.array([0, 0])]
    for a in ddp_states.root.actions:
        temp.append(numpy.array([a.RA, a.RP]) + a.next_state.VA)

    return max(temp, key=maxVA)




class action:
    def __init__(self, *args):
        if len(args) == 3:
            self.RA = args[0]
            self.RP = args[1]
            self.next_state = args[2]

class Node:
    def __init__(self):
        self.actions = []

    def add_action(self, action):
        self.actions.append(action)


class ddp:
    def __init__(self, *args):
        if len(args) == 2:
            self.root = args[0]
            self.states = args[1]




def generate_ddp(depth, num_nodes_layer, allow_layer_gap):
    root = Node()

    ddp_states = []

    for l in range(depth):
        curr = []
        for i in range(num_nodes_layer):
            vertex = Node()

            if l > 0:
                for k in range(num_nodes_layer):
                    RA = random.random()
                    RP = random.random()
                    vertex.add_action(action(RA, RP, ddp_states[l-1][k]))

            curr.append(vertex)
        ddp_states.append(curr)


    for k in range(num_nodes_layer):
        to_connect = random.random()
        if to_connect < 0.5:
            continue

        RA = random.random()
        RP = random.random()
        root.add_action(action(RA, RP, ddp_states[depth-1][k]))

    return ddp(root, ddp_states)









layers = 5
nodes_in_layer = 10
B = 1
eps_arr = [0.1, 0.05, 0.02, 0.01]
retries = 10000

graph1_mean_dfar = []
graph1_mean_bf = []
graph1_mean_cor = []
graph1_var_dfar = []
graph1_var_bf = []
graph1_var_cor = []
graph1_eps = []








for ep in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]:
    eps = ep / 1000
    graph1_eps.append(eps)
    res_dfar = []

    res_cor = []

    res_bf = []
    for i in range(retries):
        ddp_states = generate_ddp(layers, nodes_in_layer, 0)
        VA = get_VA(ddp_states, layers, nodes_in_layer)
        opt_VP = u_solver.DFAR(ddp_states, layers, nodes_in_layer, B, eps, VA)


        B_to_cor = B - (eps * layers)
        if B_to_cor < 0:
            B_to_cor = 0

        bf_opt_VP = bf_solver.solve_brute_force(ddp_states, B, VA)
        bf_cor_VP = bf_solver.solve_brute_force(ddp_states, B_to_cor, VA)

        res_bf.append(bf_opt_VP[1])
        res_cor.append(bf_cor_VP[1])
        res_dfar.append(opt_VP)

    res_bf = numpy.array(res_bf)
    res_cor = numpy.array(res_cor)

    mean_dfar = sum(res_dfar) / retries
    mean_bf = sum(res_bf) / retries
    mean_cor = sum(res_cor) / retries


    sum_diff_dfar = 0
    sum_diff_bf = 0
    sum_diff_cor = 0
    for j in range(retries):
        sum_diff_dfar += math.pow(res_dfar[j] - mean_dfar, 2)
        sum_diff_bf += math.pow(res_bf[j] - mean_bf, 2)
        sum_diff_cor += math.pow(res_cor[j] - mean_cor, 2)

    var_dfar = math.sqrt(sum_diff_dfar / (retries - 1)) / math.sqrt(retries)
    var_bf = math.sqrt(sum_diff_bf / (retries - 1)) / math.sqrt(retries)
    var_cor = math.sqrt(sum_diff_cor / (retries - 1)) / math.sqrt(retries)

    graph1_mean_dfar.append(mean_dfar)
    graph1_mean_bf.append(mean_bf)
    graph1_mean_cor.append(mean_cor)
    graph1_var_dfar.append(var_dfar)
    graph1_var_bf.append(var_bf)
    graph1_var_cor.append(var_cor)


exc_data = numpy.c_[graph1_eps, graph1_mean_bf, graph1_var_bf, graph1_mean_dfar, graph1_var_dfar]
exc_labels = ['eps', 'BF_mean', 'BF_var', 'DFAR_mean', 'DFAR_var']
df = pd.DataFrame(exc_data, columns=exc_labels)
df.to_excel('deterministic_graph1_fixed.xlsx', sheet_name='sheet1')





graph2_mean_dfar_esp = []
graph2_mean_bf = []
graph2_var_dfar = []
graph2_var_bf = []
graph2_B = []

epss = [0.01, 0.05, 0.1, 0.2, 0.5]

for Bi in range(0, 20, 1):
    B = Bi / 10
    graph2_B.append(B)
    res_dfar = []
    res_bf = []

    for i in range(retries):
        ddp_states = generate_ddp(layers, nodes_in_layer, 0)
        VA = get_VA(ddp_states, layers, nodes_in_layer)
        bf_opt_VP = bf_solver.solve_brute_force(ddp_states, B, VA)

        dfar_res_esp = []
        for eps in epss:
            opt_VP = u_solver.DFAR(ddp_states, layers, nodes_in_layer, B, eps, VA)
            dfar_res_esp.append(opt_VP)

        res_dfar.append(dfar_res_esp)
        res_bf.append(bf_opt_VP[1])

    res_dfar = numpy.array(res_dfar)
    res_bf = numpy.array(res_bf)

    mean_dfar = numpy.sum(res_dfar, axis=0) / retries
    mean_bf = numpy.sum(res_bf) / retries

    unbiased_dfar = res_dfar - numpy.array(mean_dfar)
    var_dfar = numpy.diag(numpy.sqrt(
        numpy.dot(numpy.transpose(unbiased_dfar), unbiased_dfar) / (retries - 1)
        ) / math.sqrt(retries))
    var_bf = math.sqrt(numpy.dot(res_bf - mean_bf, numpy.transpose(res_bf - mean_bf)) / (retries - 1)) / math.sqrt(retries)

    graph2_mean_dfar_esp.append(mean_dfar)
    graph2_mean_bf.append(mean_bf)
    graph2_var_dfar.append(var_dfar)
    graph2_var_bf.append(var_bf)


exc_data = numpy.c_[graph2_B, graph2_mean_bf, graph2_var_bf, graph2_mean_dfar_esp, graph2_var_dfar]
exc_labels = ['B', 'BF_mean', 'BF_var', 'DFAR_mean_001', 'DFAR_mean_005', 'DFAR_mean_01', 
              'DFAR_mean_02', 'DFAR_mean_05', 'DFAR_var_001', 'DFAR_var_005', 'DFAR_var_01', 
              'DFAR_var_02', 'DFAR_var_05']
df = pd.DataFrame(exc_data, columns=exc_labels)
df.to_excel('deterministic_graph2.xlsx', sheet_name='sheet1')
