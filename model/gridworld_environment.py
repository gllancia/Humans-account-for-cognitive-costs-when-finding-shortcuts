import numpy as np
import networkx as nx


class environment(object):
    def __init__(self,G,agent_position,goal_position,diagonal_actions=True):
        
        self.load_map(G,diagonal_actions)
        self.reset(agent_position,goal_position)
        
    
    def load_map(self,G,diagonal_actions):
        if diagonal_actions:
            self.actions = np.array([(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
        else:
            self.actions = np.array([(0,1),(1,0),(0,-1),(-1,0)])
        self.states = list(G.pos.keys())
        
        P = {state : {id_action : state for id_action,_ in enumerate(self.actions)} for state in self.states}
        P_inverse = {state_1 : {} for state_1 in self.states}
        for state in self.states:
            for neigh in G[state]:
                for id_action, action in enumerate(self.actions):
                    if all(  G.pos[neigh] == (G.pos[state] + action)  ):
                        P[state][id_action] = neigh
                        P_inverse[state][neigh] = id_action
                        
        self.P = P
        self.P_inverse = P_inverse


    def reset(self,agent,goal):
        self.agent = agent
        self.goal = goal
        self.on_goal = False
            
            
    def step(self,action_id):
        new_state = self.P[self.agent].get(action_id)
        
        if new_state == None:
            print(f"Impossible action. Move id: {action_id} State: {self.agent}")
        else:
            self.agent = new_state
            if self.agent == self.goal:
                self.on_goal = True
            else:
                self.on_goal = False
            return self.agent, self.on_goal
        
        
    def get_actions_from_state_sequence(self,sequence):
        actions = []
        state_old = sequence[0]
        for state in sequence[1:]:
            state_new = state
            action = self.P_inverse[state_old][state_new]
            
            state_old = state_new
            actions.append(action)
        return actions
    
    
    def extract_states_and_actions(self,sequence):
        """
        States encountered more than once in a row are removed
        """
        
        new_sequence = []
        for i,s in enumerate(sequence):
            if i == 0:
                new_sequence.append(s)
            else:
                if s != new_sequence[-1]:
                    new_sequence.append(s)
        actions = self.get_actions_from_state_sequence(new_sequence)
        
        return new_sequence, actions