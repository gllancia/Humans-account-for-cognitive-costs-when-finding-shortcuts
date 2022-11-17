import numpy as np
from scipy.stats import entropy
from scipy.special import softmax


    
    
def control_information_path(path, default_policy, agent_policy=None, agent_actions=None,
                             beta=None, custom_kernel=None, hist=False, diagonal_actions=True):
    """
    Inputs:
        path : 1-D list
            List of states
        default_policy : 2-D list
            Default policy, 2-D list with shape (n_states,n_actions)
        agent_policy : 2-D list
            Agent policy, 2-D list with shape (n_states,n_actions)
        agent_actions : 1-D list
            If agent policy is not supplied (i.e. experimental data), functions reconstructs policy from states and actions
        beta : float
            Softmax inverse temperature

    Returns:
        control_information
        
        agent_policy
    """
    
    control_information_history = []
    
    if agent_actions is not None:
        agent_policy = np.zeros(default_policy.shape)
        p_s =  np.zeros(default_policy.shape[0])
        
        for s,a in zip(*(path,agent_actions)):
            agent_policy[s,a] += 1
            p_s[s] += 1
        for s in range(len(p_s)):
            if p_s[s] != 0:
                agent_policy[s] = agent_policy[s]/p_s[s]
            
        
        
            
        if beta is not None:
            agent_policy = softmax(beta*agent_policy, axis=1)
        elif custom_kernel is not None:
            if custom_kernel == "default":
                agent_policy = apply_kernel(agent_policy, axis=1, diagonal_actions=diagonal_actions)
            else:
                agent_policy = apply_kernel(agent_policy, axis=1, diagonal_actions=diagonal_actions,
                                               kernel=custom_kernel)
        else:
            agent_policy += 1e-200
            default_policy += 1e-200
    

    
    for state in path[:-1]:
        control_information_history.append(entropy(agent_policy[state],default_policy[state]))
    
    control_information = sum(control_information_history)
    
    if hist == False:
        return control_information, agent_policy
    elif hist == True:
        return control_information, agent_policy, control_information_history


def apply_kernel(array,axis=0,kernel=None,diagonal_actions=True):
    if kernel is None:
        if diagonal_actions:
            kernel = [128,2,0.1,0.1,0.1,0.1,0.1,2]
        else:
            kernel = [128,4,0.5,4]
        
    
    if axis == 0:
        return _apply_kernel(array, kernel)
    elif axis == 1:
        for i in range(len(array)):
            array[i] = _apply_kernel(array[i], kernel)
        return np.array(array)
    else:
        "apply_kernel currently defined only for 1-D or 2-D arrays"

        
        
def _apply_kernel(array, kernel):
    """
    About 10-15 times faster than using numpy.roll()
    """
 
    newarray = [1e-200]*len(kernel)
    
    for i,el in enumerate(array):
        newarray = [newarray[i]+el*kernel[i] for i in range(len(kernel))]
        kernel.insert(0,kernel.pop(-1))
    newarray_sum = float(sum(newarray))
    return np.array([newarray[i]/newarray_sum for i in range(len(kernel))])


