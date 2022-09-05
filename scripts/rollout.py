import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import time
import csv
import os

import numpy as np
import tensorflow as tf
import networkx as nx
import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl

def rollout(agent,env,save_dir,rl=False,iterations=1):
    '''
    Rolls out a pre-trained DRL agent or some other packing agent on
    an episode 'iterations' number of times. Used for testing trained
    policies or other heuristics against each other on the same environment
    (same seed etc). 

    Args:

    agent (PackingAgent,ray.rllib.agents.ppo.PPOTrainer): an agent of some sort that can make node-selection
                                                            decisions in the context of the network-aware
                                                            resource allocation problem handled here.
    env (ParametricActionWrapper): an instance of an environment wrapping the PackingEnv environment modelling 
                                    resource usage in resource disaggregated data centres. Used to interface 
                                    that environment with multi-faceted observation spaces (and Rllib).
    save_dir (str): path name referring to where output data should be saved.
    rl (bool=False): whether the agent is DRL-based or not.
    iterations (int=1): how many times should the test be run.
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i in range(iterations):

        itr = '_{}'.format(i)

        #csv file initialisation preamble
        actions_file = open('{}/actions{}.csv'.format(save_dir,itr),'w')
        actions_writer = csv.writer(actions_file)
        actions_writer.writerows([['action','request_id']])
        actions = []

        acceptance_file = open('{}/acceptance{}.csv'.format(save_dir,itr),'w')
        acceptance_writer = csv.writer(acceptance_file)
        acceptance_writer.writerows([['acceptance_ratio']])
        acceptance = []

        number_of_requests = 0
        number_of_accepted = 0

        util_file = open('{}/util{}.csv'.format(save_dir,itr),'w')
        util_writer = csv.writer(util_file)
        util_writer.writerows([['compute','memory','bandwidth','compute_fragmentation','memory_fragmentation']])
        util = []

        cpu_features_file = open('{}/cpu_features{}.csv'.format(save_dir,itr),'w')
        cpu_features_writer = csv.writer(cpu_features_file)
        cpu_features_writer.writerows([['cpu']])
        cpu_features = []

        mem_features_file = open('{}/mem_features{}.csv'.format(save_dir,itr),'w')
        mem_features_writer = csv.writer(mem_features_file)
        mem_features_writer.writerows([['mem']])
        mem_features = []

        sr_bw_features_file = open('{}/sr_bw_features{}.csv'.format(save_dir,itr),'w')
        sr_bw_features_writer = csv.writer(sr_bw_features_file)
        sr_bw_features_writer.writerows([['sr']])
        sr_bw_features = []

        ra_bw_features_file = open('{}/ra_bw_features{}.csv'.format(save_dir,itr),'w')
        ra_bw_features_writer = csv.writer(ra_bw_features_file)
        ra_bw_features_writer.writerows([['ra']])
        ra_bw_features = []

        ac_bw_features_file = open('{}/ac_bw_features{}.csv'.format(save_dir,itr),'w')
        ac_bw_features_writer = csv.writer(ac_bw_features_file)
        ac_bw_features_writer.writerows([['ac']])
        ac_bw_features = []

        success_file = open('{}/success{}.csv'.format(save_dir,itr),'w')
        success_writer = csv.writer(success_file)
        success_writer.writerows([['compute','memory','bandwidth','holding_time','node_link_ratio','num_nodes']])
        success = []

        failure_file = open('{}/failure{}.csv'.format(save_dir,itr),'w')
        failure_writer = csv.writer(failure_file)
        failure_writer.writerows([['compute','memory','bandwidth','holding_time']])
        failure = []

        t = time.time()
        obs = env.reset()
        done = False

        while not done:

            current_request = env.wrapped.current_request
            
            if rl:
                action = agent.compute_action(obs)
            else:
                action = agent.compute_action(obs,env)
            
            n_shp = env.wrapped.dgl_graph.ndata['orig_node'].shape
            ndata = tf.reshape(env.wrapped.dgl_graph.ndata['orig_node'],(n_shp[0]*n_shp[1],))
            compute_features = ndata[::2]
            memory_features = ndata[1::2]

            e_shp = env.wrapped.dgl_graph.edata['orig_edge'].shape
            edata = tf.reshape(env.wrapped.dgl_graph.edata['orig_edge'],(e_shp[0]*e_shp[1],))
            bandwidth_features = edata[:]


            obs,reward,done,info = env.step(action)

            #record action taken at this step
            req_id = int(current_request.split('_')[1])
            actions.append([action,req_id])
            
            if reward == 10 or reward == -10:

                cpu_features.append([env.wrapped.manager.network.components[node].available['node']/env.wrapped.manager.network.components[node].maximum['node'] for node in env.wrapped.manager.network.components_of_sub_type('Resource')])
                cpu_features_writer.writerows(cpu_features)
                cpu_features = []

                mem_features.append([env.wrapped.manager.network.components[node].available['mem']/env.wrapped.manager.network.components[node].maximum['mem'] for node in env.wrapped.manager.network.components_of_sub_type('Resource')])
                mem_features_writer.writerows(mem_features)
                mem_features = []

                sr_bw_features.append([env.wrapped.manager.network.components[node].available['ports']/env.wrapped.manager.network.components[node].maximum['ports'] for node in env.wrapped.manager.network.components_of_sub_type('SRLink')])
                sr_bw_features_writer.writerows(sr_bw_features)
                sr_bw_features = []

                ra_bw_features.append([env.wrapped.manager.network.components[node].available['ports']/env.wrapped.manager.network.components[node].maximum['ports'] for node in env.wrapped.manager.network.components_of_sub_type('RALink')])
                ra_bw_features_writer.writerows(ra_bw_features)
                ra_bw_features = []

                ac_bw_features.append([env.wrapped.manager.network.components[node].available['ports']/env.wrapped.manager.network.components[node].maximum['ports'] for node in env.wrapped.manager.network.components_of_sub_type('ACLink')])
                ac_bw_features_writer.writerows(ac_bw_features)
                ac_bw_features = []

                print('request: {}'.format(number_of_requests))
                req = env.wrapped.manager.allocated_requests[current_request]

                #iterate total number of requests for acceptance ratio
                number_of_requests += 1

                #record utilisation information
                util.append([info['util'][0],info['util'][1],info['net_util'],info['fragmentation'][0],info['fragmentation'][1]])

                if reward == 10:

                    #iterate number_of_accepted for acceptance ratio record
                    number_of_accepted += 1

                    #write request-requirement values and link:node ratio to success
                    nodes = 0
                    edges = 0

                    for comp in info['allocated']:
                        if env.wrapped.manager.network.components[comp].type == 'Edge':
                            edges += 1
                        if env.wrapped.manager.network.components[comp].type == 'Node':
                            nodes += 1

                    success.append([req.requirements['node'],
                                    req.requirements['mem'],
                                    req.requirements['ports'],
                                    req.requirements['holding_time'],
                                    edges/nodes,
                                    nodes]
                                    )

                if reward == -10:

                    #write request-requirement values to failure
                    failure.append([req.requirements['node'],
                                    req.requirements['mem'],
                                    req.requirements['ports'],
                                    req.requirements['holding_time']])

                #record acceptance ratio after this request
                acceptance.append([number_of_accepted/number_of_requests])
                print('acceptance: {}'.format(number_of_accepted/number_of_requests))

                #write to and close all csv files
                actions_writer.writerows(actions)
                acceptance_writer.writerows(acceptance)
                util_writer.writerows(util)
                success_writer.writerows(success)
                failure_writer.writerows(failure)

                actions = []
                acceptance = []
                util = []
                success = []
                failure = []

        actions_file.close()
        acceptance_file.close()
        util_file.close()
        success_file.close()
        failure_file.close()

        cpu_features_file.close()
        mem_features_file.close()
        sr_bw_features_file.close()
        ra_bw_features_file.close()
        ac_bw_features_file.close()
    
class PackingAgent:
    '''
    Class implementing heuristics such that their action-taking interface
    corresponds with the action-taking interfaces of restored Rllib Trainer
    instances. This allows different heuristics to be used in the same 
    scripts as a trained Rllib Trainer instance.
    '''
    def __init__(self,packing_heuristic,cluster_size=None):
        '''
        Args:

        packing_heuristic (str): string referring to the name of a particular heuristic 
                                    ('tetris','nulb','nalb',random' are supported and tested).
        '''
        self.packing_heuristic = packing_heuristic

        if self.packing_heuristic == 'tetris':
            self.last_action = None
        if self.packing_heuristic == 'nulb':
            self.last_action = None
            self.action_series = []
            self.current_request = ''
        if self.packing_heuristic == 'nalb':
            self.last_action = None
            self.action_series = []
            self.current_request = ''

    def compute_action(self,obs,env):

        if self.packing_heuristic == 'best_fit':
            env.wrapped.lb_route_weighting = False
            return self.best_fit(obs,env)
        if self.packing_heuristic == 'random':
            env.wrapped.lb_route_weighting = False
            return self.random(obs,env)
        if self.packing_heuristic == 'random_clustering':
            env.wrapped.lb_route_weighting = False
            return self.random_clustering(obs,env)
        if self.packing_heuristic == 'tetris':
            env.wrapped.lb_route_weighting = False
            return self.tetris(obs,env)
        if self.packing_heuristic == 'nulb':
            env.wrapped.lb_route_weighting = True
            return self.nulb(obs,env)
        if self.packing_heuristic == 'nalb':
            env.wrapped.lb_route_weighting = True
            return self.nalb(obs,env)
    

    def random(self,obs,env):
        '''
        Implements a random node selection policy.
        '''
        valid_actions = np.where(np.array(obs['action_mask']) != 0)[0]

        if len(valid_actions) > 0:
            idx = np.random.choice(valid_actions)
            return idx
        else:
            return -1
            
        
    def tetris(self,obs,env):
        '''
        This implements the Tetris algorithm from literature, applied in the 
        context of network aware allocation in disaggregate data centres:

        https://dl.acm.org/doi/10.1145/2619239.2626334
        '''
        #get normalised node and edge features from dgl_graph with no scaling
        dgl_graph = env.wrapped.manager.network.to_dgl_with_edges(env.wrapped.features)

        edges = tf.math.multiply(dgl_graph.edata['features'],1/1)
        edge_ids = tf.cast(env.wrapped.dgl_graph.edges()[0],tf.float32)

        nodes = tf.math.multiply(dgl_graph.ndata['features'],1/10)
        all_node_idx = tf.ones(obs['action_mask'].shape)
        valid_nodes = tf.where(obs['action_mask'] == 1)

        valid_edges = [edge.numpy() in valid_nodes.numpy() for edge in edge_ids]
        valid_edges = tf.where(valid_edges)

        edge_features = tf.reshape(tf.gather(edges,valid_edges),[valid_edges.shape[0], 1])
        edge_features_zeros = tf.zeros((nodes.shape[0]-edge_features.shape[0],1))
        edge_features = tf.concat([tf.cast(edge_features_zeros,tf.float32), tf.cast(edge_features,tf.float32)],0)
        
        features = tf.concat([nodes, edge_features],-1)

        req = env.wrapped.manager.allocated_requests[env.wrapped.current_request].requirements
        req_vec = tf.convert_to_tensor([req[ft]/10 for ft in env.wrapped.features['node']]+[req[ft]/1 for ft in env.wrapped.features['link']],dtype=tf.float32)
        
        #get dot product and apply masking.
        dot_prod = tf.tensordot(tf.math.l2_normalize(features),tf.math.l2_normalize(req_vec),1)

        scores = tf.math.multiply(tf.cast(dot_prod,tf.float32),tf.cast(obs['action_mask'],tf.float32),1)

        #apply locality penalty if first choice has already been made
        if self.last_action is not None:
            source_node = env.wrapped.dgl_graph.ndata['_name'][self.last_action]
            source_node = str(np.char.decode(source_node.numpy()))
            neighbours = set()
            for count, k_edge in enumerate(nx.bfs_edges(env.wrapped.manager.network.graph, source=source_node, depth_limit=2)):
                neighbours.add(k_edge[0])
                neighbours.add(k_edge[1])

            neighbours = [str(np.char.decode(env.wrapped.dgl_graph.ndata['_name'].numpy()[i])) in neighbours for i in range(env.wrapped.num_actions)]
            multiplier = tf.cast([9/10 if i == False else 1 for i in neighbours],tf.float32)

            scores = tf.math.multiply(scores,multiplier,1)

        self.last_action = np.argmax(scores)

        return self.last_action

    def nulb(self,obs,env):
        '''
        This implements the NULB algorithm from literature:

        https://opg.optica.org/jocn/abstract.cfm?uri=jocn-10-2-a270
        '''
        netx = env.wrapped.manager.network.netx_lb_routing_weights()

        if not (env.wrapped.current_request == self.current_request):
            self.action_series = []
        
        #if action series has not already been chosen, and previous request did not fail mid-allocation... 
        if self.action_series == []:
            self.current_request = env.wrapped.current_request

            required = np.array([env.wrapped.manager.allocated_requests[env.wrapped.current_request].requirements[ft] for ft in env.wrapped.features['node']])

            #allocate first node using best-fit algorithm
            self.action_series.append(self.tetris(obs,env))
            component_id = env.wrapped.dgl_graph.ndata['_name'][self.action_series[-1]]
            node = str(np.char.decode(component_id.numpy()))

            #multi-resource
            amount = np.array([env.wrapped.manager.network.components[node].available[ft] for ft in env.wrapped.features['node']])

            #multi-resource
            to_allocate = np.minimum(amount,required)
            total_allocated = np.copy(to_allocate)

            #initialise mBFS loop variables
            q = [self.action_series[-1]]
            already_visited = []
            already_allocated = [node]

            while q != []:

                current = q.pop()
                
                component_id = env.wrapped.dgl_graph.ndata['_name'][current]
                source_node = str(np.char.decode(component_id.numpy()))

                already_visited.append(current)

                #get bandwidth list of current BFS source-node
                edges = [edges[-1] for edges in enumerate(nx.bfs_edges(env.wrapped.manager.network.graph, source=source_node, depth_limit=1))]

                #until visited every neighbour of the current source-node
                while edges:

                    #if first choice fully allocates, pop-return an action from self.action_series
                    if np.prod(total_allocated == required) == 1:
                        action = self.action_series.pop()
                        return action

                    #otherwise, loop over neighbours of source-node from high to low bandwidth
                    node = edges[0][-1]
                    
                    del(edges[0])
                    
                    #multi-resource
                    amount = np.array([env.wrapped.manager.network.components[node].available[ft] for ft in env.wrapped.features['node']])

                    #if some resource is present at that node, allocate resource from it
                    if np.sum(amount) > 0 and not (node in already_allocated):
                        
                        #multi-resource
                        to_allocate = np.minimum(amount,required-total_allocated)
                        total_allocated += to_allocate

                        action = [str(np.char.decode(env.wrapped.dgl_graph.ndata['_name'].numpy()[i])) == node for i in range(env.wrapped.num_actions)]
                        idx = tf.squeeze(tf.where(action)).numpy()
                        self.action_series.append(idx)
                        already_allocated.append(node)

                    #otherwise, get index corresponding to current node and add to queue (if hasn't already been checked, to avoid loops)
                    else:
                        action = [str(np.char.decode(env.wrapped.dgl_graph.ndata['_name'].numpy()[i])) == node for i in range(env.wrapped.num_actions)]
                        idx = tf.squeeze(tf.where(action)).numpy()

                        if not (idx in already_visited) and not (idx in q) and env.wrapped.manager.network.components[node].sub_type != 'Resource':
                            q.append(idx)
        else:
            action = self.action_series.pop()
            return action
        
        print('passed all loops')

    def nalb(self,obs,env):
        '''
        This implements the NALB algorithm from literature:

        https://opg.optica.org/jocn/abstract.cfm?uri=jocn-10-2-a270
        '''
        resource_nodes = env.wrapped.manager.network.components_of_sub_type('Resource')
        # print([env.wrapped.manager.network.components[comp].available['node'] for comp in resource_nodes])

        if not (env.wrapped.current_request == self.current_request):
            self.action_series = []

        if self.action_series == []:
            self.current_request = env.wrapped.current_request

            #get required amount
            required = np.array([env.wrapped.manager.allocated_requests[env.wrapped.current_request].requirements[ft] for ft in env.wrapped.features['node']])

            #allocate first node using best-fit algorithm
            self.action_series.append(self.tetris(obs,env))
            component_id = env.wrapped.dgl_graph.ndata['_name'][self.action_series[-1]]
            node = str(np.char.decode(component_id.numpy()))
            amount = np.array([env.wrapped.manager.network.components[node].available[ft] for ft in env.wrapped.features['node']])
            
            #multi-resource
            to_allocate = np.minimum(amount,required)

            total_allocated = to_allocate

            q = [self.action_series[-1]]
            already_visited = []
            already_allocated = [node]

            while q != []:

                current = q.pop()
                
                component_id = env.wrapped.dgl_graph.ndata['_name'][current]
                source_node = str(np.char.decode(component_id.numpy()))

                already_visited.append(current)

                #get bandwidth list of current BFS source-node
                edges = [edges[-1] for edges in enumerate(nx.bfs_edges(env.wrapped.manager.network.graph, source=source_node, depth_limit=1))]

                graph = env.wrapped.manager.network.graph
                bw = [env.wrapped.manager.network.components[graph.get_edge_data(edge[0],edge[1])['label']].available['ports'] for edge in edges]

                #until visited every neighbour of the current source-node
                while edges:

                    #if first choice fully allocates, pop-return an action from self.action_series
                    if np.prod(total_allocated == required) == 1:
                        action = self.action_series.pop()
                        return action

                    #otherwise, loop over neighbours of source-node from high to low bandwidth
                    idx = np.argmax(bw)
                    node = edges[idx][-1]

                    del(edges[idx])
                    del(bw[idx])

                    #multi-resource
                    amount = np.array([env.wrapped.manager.network.components[node].available[ft] for ft in env.wrapped.features['node']])

                    #if some resource is present at that node, allocate resource from it
                    if np.sum(amount) > 0 and not (node in already_allocated):
                        
                        #multi-resource
                        to_allocate = np.minimum(amount,required-total_allocated)
                        total_allocated += to_allocate

                        action = [str(np.char.decode(env.wrapped.dgl_graph.ndata['_name'].numpy()[i])) == node for i in range(env.wrapped.num_actions)]
                        idx = tf.squeeze(tf.where(action)).numpy()
                        self.action_series.append(idx)
                        already_allocated.append(node)

                    #otherwise, get index corresponding to current node and add to queue (if hasn't already been checked, to avoid loops)
                    else:
                        action = [str(np.char.decode(env.wrapped.dgl_graph.ndata['_name'].numpy()[i])) == node for i in range(env.wrapped.num_actions)]
                        idx = tf.squeeze(tf.where(action)).numpy()

                        if not (idx in already_visited) and not (idx in q) and env.wrapped.manager.network.components[node].sub_type != 'Resource':
                            q.append(idx)
        else:
            action = self.action_series.pop()
            return action

        print('passed all loops')