import numpy as np
import copy
import networkx as nx


class infoRL(object):
    def __init__(self, 
                 beta, 
                 graph, 
                 goal,
                 default_policy="random",
                 F_threshold=1e-4,
                 F_window=30,
                 r=1.,
                 diagonal_actions=True,
                 relabel_nodes=False):
        
        self.beta = beta
        self.G = copy.deepcopy(graph)
        self.goal = goal
        
        if relabel_nodes:
            self.reinitialize_nodes()
            
        self.coord2node = {(x,y):node for node,(x,y) in self.G.pos.items()}

        self.initialize_actions(diagonal_actions)
        self.initialize_default_policy(default_policy)
        self.initialize_reward_matrix(r)
        self.initialize_transition_distribution()
        
        self.generate_policy(F_threshold,F_window)

        
        
    
    def reinitialize_nodes(self):
        """
        Relabels nodes in G (edges and G.pos) from 0 to len(G).
        """
        
        relabel_dic = {node:index for index,node in enumerate(sorted(self.G))}
        self.G = nx.relabel_nodes(self.G,relabel_dic)
        self.G.pos = {relabel_dic[node]:[x,y] for node,(x,y) in self.G.pos.items()}
        self.goal = relabel_dic[self.goal]
        
        
        
        
    def initialize_actions(self,diagonal_actions):
        for node in self.G:
            self.G.pos[node] = np.array(self.G.pos[node])
            
        aux_coord2node = lambda xy : self.coord2node.get(tuple(list(xy)))


        up = lambda node : aux_coord2node(self.G.pos[node]+[0,1])
        right = lambda node : aux_coord2node(self.G.pos[node]+[1,0])
        down = lambda node : aux_coord2node(self.G.pos[node]+[0,-1])
        left = lambda node : aux_coord2node(self.G.pos[node]+[-1,0])

        if diagonal_actions == True:
            up_right = lambda node : aux_coord2node(self.G.pos[node]+[1,1])
            down_right = lambda node : aux_coord2node(self.G.pos[node]+[1,-1])
            down_left = lambda node : aux_coord2node(self.G.pos[node]+[-1,-1])
            up_left = lambda node : aux_coord2node(self.G.pos[node]+[-1,1])

            self.actions = [up,up_right,right,down_right,down,down_left,left,up_left] #actions return None if end up outside map
        else:
            self.actions = [up,right,down,left]

            
                       
                
    def initialize_default_policy(self,default_policy):
        if type(default_policy) == str:
            if default_policy == "random":
                self.rho_s_a = np.ones((len(self.G),len(self.actions)))
                self.rho_s_a /= np.expand_dims(self.rho_s_a.sum(axis=1),1)
            else:
                print("Specify a default policy.")
        else:
            self.rho_s_a = copy.deepcopy(default_policy)
            
            
            
            
    def initialize_reward_matrix(self, r):
        self.R_s_a = -r * 10 * np.ones((len(self.G),len(self.actions)))
        for s1 in self.G.nodes:
            for action_index,action in enumerate(self.actions):
                s2 = action(s1)
                if s2 in self.G[s1]:
                    self.R_s_a[s1,action_index] = -r

        self.R_s_a[self.goal] = np.zeros(len(self.actions))


        
        
    def initialize_transition_distribution(self):
        self.p_s_s_a = np.zeros((len(self.G),len(self.G),len(self.actions)))
        for s1 in self.G.nodes:
            for action_index,action in enumerate(self.actions):
                s2 = action(s1)
                if s2 in self.G[s1]:
                    self.p_s_s_a[s2,s1,action_index] = 1.
                else:
                    self.p_s_s_a[s1,s1,action_index] = 1.

        for s in self.G.nodes:
            for action_index in range(len(self.actions)):
                self.p_s_s_a[s,self.goal,action_index] = 0.

        self.p_s_s_a[self.goal,self.goal] = np.ones(len(self.actions))


        
    
    def generate_policy(self,F_threshold,F_window,max_iterations=5000):
        F_s = np.zeros(len(self.G))
        
        F_hist = [F_s]
        F_deriv_hist = []
        
        for it in range(max_iterations):
            Q_F_s_a = np.dot(F_s,np.transpose(self.p_s_s_a,(1,0,2))) - self.beta*self.R_s_a
            Z_s = np.array([np.dot(self.rho_s_a[s],np.exp(-Q_F_s_a[s]).T) for s in self.G.nodes])
            F_s = -np.log(Z_s)
            
            F_hist = np.append(F_hist,[F_s],axis=0)
            F_deriv_hist.append(np.mean(np.abs(F_hist[-1]-F_hist[-2])))
            if it > F_window:
                if np.mean(F_deriv_hist[-F_window:]) < F_threshold:
                    break
            if it == max_iterations-1:
                print("Maximum number of iterations reached.")
                
        self.F_hist = F_hist[:]
        self.pi_s_a = self.rho_s_a*np.exp(-Q_F_s_a)/np.tile(Z_s,(len(self.actions),1)).T