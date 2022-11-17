import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt

from scipy.special import softmax



def get_RGI(G,beta=10.,n_steps=50,diagonal_actions=False,distances=None):
    
    n_states = len(G)
    
    coord2ind = {(x,y):node for node,(x,y) in G.pos.items()}
        
    up = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[0,1])))
    up_right = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[1,1])))
    right = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[1,0])))
    down_right = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[1,-1])))
    down = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[0,-1])))
    down_left = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[-1,-1])))
    left = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[-1,0])))
    up_left = lambda node : coord2ind.get(tuple(list(np.array(G.pos[node])+[-1,1])))
        
        

    if diagonal_actions:
        actions = [up,up_right,right,down_right,down,down_left,left,up_left]
    else:
        actions = [up,right,down,left]
    n_actions = len(actions)
    
    if distances is None:
        #generate Q
        distances = {node:{} for node in G.nodes}
        for node1 in G.nodes:
            d = nx.algorithms.shortest_path_length(G,node1)
            for node2 in G.nodes:
                distances[node1][node2] = d[node2]
                distances[node2][node1] = d[node2]
    
    Q = np.zeros((n_states,n_states,n_actions))
    node2i = {node:i for i,node in enumerate(list(G.nodes))}
    for n1 in G.nodes:
        for goal in G.nodes:
            scores = []
            scores_index = []
            for i,a in enumerate(actions):
                n2 = a(n1)
                if n2 in list(G[n1]):
                    scores.append(-distances[goal][n2])
                    scores_index.append(i)
            scores = softmax(10.*np.array(scores))

            for s,s_i in zip(scores,scores_index):
                Q[node2i[goal],node2i[n1],s_i] = s
    
    
    #generate uniform p_a_s
    p_a_s = np.where(Q.mean(axis=0)>0., 1.,0.)
    p_a_s = p_a_s / p_a_s.sum(axis=1).reshape((p_a_s.shape[0],1))


    history_pasg = []
    for _ in range(n_steps):
        #generate p_a_s_g
        p_a_s_g = np.array([p_a_s for _ in range(n_states)])*np.exp(beta*Q)
        temp = p_a_s_g.sum(axis=2)
        Z = np.stack((temp,temp,temp,temp),axis=2)

        p_a_s_g = p_a_s_g / Z
        history_pasg.append(np.abs(p_a_s_g))

        #generate p_a_s
        p_a_s = p_a_s_g.mean(axis=0)
    history_pasg = np.array(history_pasg)
        
    p_a_s += 10**-200
    I1 = p_a_s * np.log(p_a_s)
    I1 = I1.sum(axis=-1)
    
    p_a_s_g += 10**-200
    I2 = p_a_s_g * np.log(p_a_s_g)
    I2 = I2.mean(axis=0).sum(axis=-1)
    
    I = -I1 + I2
    
    return I, history_pasg