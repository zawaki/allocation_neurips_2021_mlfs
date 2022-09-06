# """
# Trivial gym environment that will be used to sanity check understanding of gym/rllib.

# Reward is defined by avg. utilization per server node w.r.t. their single compute resource ('node').

# Since greedy w.r.t. high utilization corresponds directly to a best-fit heuristic, expect a q-learning
# algorithm (implemented with the generic rllib framework for it) to learn a best fit over time.

# For now, will allow the framework to also choose switch nodes, to see if it can reliably learn to ignore
# all nodes which provide none of the requested resource.

# Network dynamics/allocations are totally ignored in this sanity-check example.
# """

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

import gym
from gym import error, spaces, utils
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding

import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl
from sklearn import preprocessing
from math import inf
import random as rnd
import numpy as np
from itertools import islice

from dc_network import *

from gnn import *


NUM_FEATURES = 2
OBS_SPACE_DIM = 2
EMBEDDING_DIM = 1

class PackingEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, env_config):

        super(PackingEnv, self).__init__()

        self.seed = env_config['rnd_seed']
        rnd.seed(self.seed)

        print('initialising environment')
        
        self.num_init_requests = env_config['num_init_requests']
        self.network_dir = env_config['network_dir']
        self.features = env_config['features']
        self.request_type = env_config['request_type']
        #lb methods
        self.lb_route_weighting = env_config['lb_route_weighting']

        self.seed_on_reset = env_config['seed_on_reset']
        self.reset()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=-10000,high=10000,shape=(len(self.features['node'])+1,))
        
        self.path_links = set()

        print('initialised')

    def reset(self):
        '''
        Reset environment: re-initialise graph (de-allocate everything); 
        initialise the new first request; return the observation of the new environment.
        '''
        self.time = 0
        if self.seed_on_reset:
            rnd.seed(self.seed)

        self.dropped = 0
        self.successes = 0
        self.failed = 0

        self.curr_req_chosen = []

        #metric variables
        self.allocated_requests = 0

        '''init manager'''
        self.manager = NetworkManager(self.network_dir)

        '''get DGL graph from manager'''
        self._get_graph()

        #multi-resource
        # self.initial_resource_quantity = np.sum(self.dgl_graph.ndata['features'],axis=0)[0]
        self.initial_resource_quantity = np.array([np.sum(self.dgl_graph.ndata['features'],axis=0)[ft] for ft in range(len(self.features['node']))])
        self.total_resource_amount = np.copy(self.initial_resource_quantity)
        self.path_links = set()
        self.num_actions = len(self.dgl_graph.ndata['_ID'])
        
        self.request_iter = 0
        self._update_current_request()
        
        return self._get_observation()

    def connectivity_check(self,component_id, k=3):
        '''
        Given a selected server, check if sufficient netwrok resources exist
        within the network such that all-to-all connectivity can be established
        between that server and all other servers that have currently been
        allocatd to the current request.

        Uses k-shortest paths to check that ports are available along
        paths between servers.

        Args:

        component_id (dict): string refering to the name of a server to be considered
                                for allocation.
        k (int): how many paths to search over within the k-shortest paths procedure.
        '''

        bool_1 = False
        bool_2 = False

        if (self.manager.allocated_requests[self.current_request].requirements['node'] == self.manager.allocated_requests[self.current_request].allocated['node']) or self.manager.network.components[component_id].available['node'] == 0.0:
            bool_1 = True

        components = [comp for comp in list(self.manager.allocated_requests[self.current_request].allocations.keys()) if self.manager.network.components[comp].sub_type == 'Resource']
    

        all_links = {}
        failed = False

        #lb methods
        if not self.lb_route_weighting:
            graph = self.manager.network.graph
            weight = None
        else:
            graph = self.manager.network.netx_lb_routing_weights()
            weight = 'weight'

        for comp in components:

            if self.manager.network.components[comp].allocations[self.current_request]['node'] == 0.0:
                bool_2 = True

            if bool_1 and bool_2:
                continue

            paths = list(islice(nx.shortest_simple_paths(graph,component_id,comp,weight=weight), k))

            for path in paths:
                failed = False
                path_links = {}

                for i in range(1,len(path)):
                    link = self.manager.network.graph.get_edge_data(path[i-1],path[i])['label']
                    ports = self.manager.network.components[link].available['ports']

                    if ports < 1:
                        failed = True
                    else:
                        if link in path_links.keys():
                            path_links[link] += 1
                        else:
                            path_links[link] = 1

                if not failed:
                    
                    for lnk in path_links.keys():
                        if lnk in all_links.keys():
                            all_links[lnk] += 1
                        else:
                            all_links[lnk] = 1     
                    
                    self._allocate_network(path_links)

                    break   

            
            if failed:
                return failed
            
        return failed


    def _allocate_network(self,path_links):
        '''
        Allocate a specific number of ports from each link
        in the network, specified by link_dict.

        Args:

        path_links (dict): dictionary where keys = names of links in the network &
                            values = how many ports to remove (allocate) from that link.
        '''

        for link,ports_to_allocate in path_links.items():

            self.manager.allocate(self.current_request,link,'ports',ports_to_allocate)
            self.links_to_allocate = {}

    def _update_request_size_checking(self):
        '''
        Fetch the next request. If not enough resources are available to conceivably
        allocate that request, then let time pass until enough requests have de-allocated
        for the allocation to be possible.
        '''
        self.curr_req_chosen = []
        self._update_current_request()
        self._check_holding_times()

        check = 0

        req_vec = np.array([self.manager.allocated_requests[self.current_request].requirements[ft] for ft in self.features['node']])

        while np.any(self.initial_resource_quantity < req_vec):
            self._check_holding_times()

        self._get_graph()
    
    def step(self, action):
        '''
        Attempt to allocate resources from a server (specified by action) to the
        current request. High level summary of procedure is as follows:

        1. If there are insufficient network resources between the selected server
        and previously selected servers (for the same request), fail:
            a. drop the request
            b. reset environment and return state and reward.
        2. Else if networking resources can be found and the current request is now
        fully provisioned:
            a. allocate all server and networking resources found for that request
            to that request uniquely (i.e. remove them from the resource pool until 
            that request has fulfiled it's holding time)
            b. check if episode is done.
        3. Else if networking resources can be found but the current request is not 
        yet fully provisioned (i.e. more servers need to be selected):
            a. do not allocate selected server and network resources yet (but keep a record
            of the newly added server and network resources from the most recent action).
            b. move on to another action
        4. return state, reward etc

        Args:

        action (int): number that refers uniquely to a specific server in the data centre.
        '''

        allocated_comps = None
        self.curr_req_chosen.append(action)

        component_id = self.dgl_graph.ndata['_name'][action]
        component_id = str(np.char.decode(component_id.numpy()))

        failed = self.connectivity_check(component_id)
        if failed:
                
            self.manager.de_allocate(self.current_request)
            self.failed += 1
            self.manager.allocated_requests[self.current_request].requirements['allocated'] = False

            if self._done():

                done = True
                reward = self._get_reward(failed=failed)
                observation = self._get_observation()
                info = {'successes':self.successes,
                        'failed':self.failed,
                        'dropped':self.dropped,
                        'util':1-self.initial_resource_quantity/self.total_resource_amount,
                        'allocated':allocated_comps,
                        'net_util':self._network_util(),
                        'rack_net_util':self._rack_net_util(),
                        'fragmentation':self._server_fragmentation(),
                        'servers_used':self._percent_servers_used()}

                return observation,reward,done,info
            
            else:
                self.manager.de_allocate(self.current_request)
                # self.path_links = set()
        
                done = self._done()
                if not done:
                    self._update_request_size_checking()

                observation = self._get_observation()
                
                reward = self._get_reward(failed=failed)
                info = {'successes':self.successes,
                        'failed':self.failed,
                        'dropped':self.dropped,
                        'util':1-self.initial_resource_quantity/self.total_resource_amount,
                        'allocated':allocated_comps,
                        'net_util':self._network_util(),
                        'rack_net_util':self._rack_net_util(),
                        'fragmentation':self._server_fragmentation(),
                        'servers_used':self._percent_servers_used()} 

                return observation,reward,done,info

        else:
            for resource in self.features['node']:

                available_quantity = self.manager.network.components[component_id].available[resource]
                required_quantity = self.manager.allocated_requests[self.current_request].requirements[resource]
                allocated_quantity = self.manager.allocated_requests[self.current_request].allocated[resource]
                to_allocate = min(available_quantity,required_quantity-allocated_quantity)

                self.manager.allocate(self.current_request,component_id,resource,to_allocate)

        observation = self._get_observation()

        requested_node = [self.manager.allocated_requests[self.current_request].requirements[resource] for resource in self.features['node']]
        allocated_node = [self.manager.allocated_requests[self.current_request].allocated[resource] for resource in self.features['node']]

        self.successful_allocation = (requested_node == allocated_node)
        if self.successful_allocation:

            self.successes += 1
            self.manager.allocated_requests[self.current_request].requirements['allocated'] = True
            self.manager.allocated_requests[self.current_request].requirements['allocation_time'] = self.time
            self.allocated_requests += 1

            req_vec = np.array([self.manager.allocated_requests[self.current_request].requirements[ft] for ft in self.features['node']])
            self.initial_resource_quantity -= req_vec

            allocated_comps = list(self.manager.allocated_requests[self.current_request].allocations.keys())

            if self.request_iter == self.num_init_requests:
                done = True
            else:
                done = False
                if not done:
                    self._update_request_size_checking()

        else:
            done = False
            self._check_holding_times(time_increment=False)
            self._get_graph()

        reward = self._get_reward(allocated=self.successful_allocation)
        
        info = {'successes':self.successes,
                'failed':self.failed,
                'dropped':self.dropped,
                'util':1-self.initial_resource_quantity/self.total_resource_amount,
                'allocated':allocated_comps,
                'net_util':self._network_util(),
                'rack_net_util':self._rack_net_util(),
                'fragmentation':self._server_fragmentation(),
                'servers_used':self._percent_servers_used()}

        return observation,reward,done,info

    def _done(self):
        '''
        Episode is done if the specified number of requests have been received
        (whether they were successfully or unsuccessfully allocated does not matter).
        '''
        return self.request_iter == self.num_init_requests

    def _get_observation(self):
        '''
        Return the (global component of) the environment state.

        This consists of the utilisation of each resource type and the holding-time
        of the current request.
        '''
        packing_eff = 1-(self.initial_resource_quantity/self.total_resource_amount)
        cur_req_hold_time = self.manager.allocated_requests[self.current_request].requirements['holding_time_feat']

        return np.append(packing_eff,cur_req_hold_time)

    def _network_util(self):
        '''
        Calculate and return the utilisation of ports in the whole data centre network.
        '''
        net_utl = 0
        links = self.manager.network.components_of_type('Edge')
        for link in links:
            net_utl += (1 - self.manager.network.components[link].available['ports']/self.manager.network.components[link].maximum['ports'])

        return net_utl/len(links)

    def _rack_net_util(self):
        '''
        Calculate and return the utilisation of only rack (tier-1) links in the whole data centre network.
        '''
        net_utl = 0
        links = self.manager.network.components_of_sub_type('SRLink')
        for link in links:
            net_utl += (1 - self.manager.network.components[link].available['ports']/self.manager.network.components[link].maximum['ports'])

        return net_utl/len(links)

    def _server_fragmentation(self):
        '''
        Calculate and return the resource fragmentation of servers in the whole data centre.
        '''
        fragmentation = np.zeros(len(self.features['node']))
        ones = np.ones(len(self.features['node']))
        num_servers = 0
        servers = self.manager.network.components_of_type('Node')
        for server in servers:
            if self.manager.network.components[server].allocations != {}:
                available =  np.array([self.manager.network.components[server].available[ft] for ft in self.features['node']])
                maximum =  np.array([self.manager.network.components[server].maximum[ft] for ft in self.features['node']])
                fragmentation += (ones - available/maximum)
                num_servers += 1
        
        return fragmentation/num_servers if num_servers > 0 else fragmentation

    def _percent_servers_used(self):

        used = 0
        servers = self.manager.network.components_of_type('Node')
        for server in servers:
            if self.manager.network.components[server].allocations != {}:
                used += 1
        

    def _update_current_request(self):
        '''
        Fetch a new request and make this the new current request.

        Request is chosen randomly from a pre-generated list of
        randomly generated requests.
        '''
        self.request_iter += 1
        self.manager.get_request(self.request_type)
        self.new_req = True

        self.current_request = np.random.choice(list(self.manager.buffered_requests.keys()))
        self.manager.move_request_to_allocated(self.current_request)


    def _check_holding_times(self,time_increment=True):
        '''
        Check if any requests have (at the current time increment) reached their
        current holding period. If so, de-allocate their resources.

        Args:

        time_increment (bool): If True, the time will automatically iterate by 1 unit
                                after this function has checked once for de-allocations.
        '''
        for req in self.manager.allocated_requests.keys():
            if self.manager.allocated_requests[req].requirements['allocated'] == True:
                if (self.time - self.manager.allocated_requests[req].requirements['allocation_time']) == self.manager.allocated_requests[req].requirements['holding_time']:

                    self.manager.de_allocate(req)
                    self.manager.allocated_requests[req].requirements['allocated'] = False

                    #multi-resource
                    req_vec = np.array([self.manager.allocated_requests[req].requirements[ft] for ft in self.features['node']])
                    # self.initial_resource_quantity += self.manager.allocated_requests[req].requirements['node']
                    self.initial_resource_quantity += req_vec
        
        if time_increment:
            self.time += 1

    def _get_reward(self,allocated=False,failed=False):
        '''
        Return the appropriate reward based on whether the request
        has 1. failed, 2. not failed but not yet allocated and 
        3. not failed and is allocated.

        Args:

        allocated (bool=False): True if the current request has been provisioned
                                    as many resources as it initially requested.
        failed (bool=False): True if networking resources could not be found for
                                the most recently selected server for the current
                                request.
        '''
        if allocated:
            return 10
        elif failed:
            return -10
        else:
            return 0

    def _get_graph(self):
        '''
        Initialises a DGL.Graph form of the graph that is defined by the 
        NetworkManager class. This is so that it can be passed to/used by
        the graph neural network implementation.
        '''
        self.dgl_graph = self.manager.network.to_dgl_with_edges(self.features)
        

class ParametricActionWrapper(gym.Env):
    '''
    This class serves as a wrapper around the basic environment (which itself implements
    the fundamental resource allocation actions and records). The purpose of this wrapper 
    is so that a multi-faceted observation can be passed from the environment to the 
    policy model. This is done in line with the recommendations of how to do this when 
    using the Rllib library.
    '''
    def __init__(self, env_config):

        self.wrapped = PackingEnv(env_config)

        num_actions = self.wrapped.action_space.n
        num_edges = self.wrapped.dgl_graph.number_of_edges()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Dict({
            "avail_actions": spaces.Box(-10, 10, shape=(num_actions, len(self.wrapped.features['node']))),
            "action_mask": spaces.Box(0, 1, shape=(num_actions,)),
            "network": self.wrapped.observation_space,
            "edges":spaces.Box(-100, 100, shape=(num_edges, len(self.wrapped.features['link']))),
            "chosen_nodes":spaces.Box(0, 1, shape=(num_actions,)),
            "edge_ids":spaces.Box(0,num_actions, shape=(num_edges,))
        })
        self.chosen = np.zeros(num_actions)

    def update_avail_actions(self):
        '''
        This function prepares/updates a set of vectors containing information about
        the (normalised) resource availability of nodes and links in the data centre.

        Resources are normalised with respect to the currently request quantity.

        These vectors are then passed (along with topological information etc) to
        the graph neural network component, which can then construct a DGL graph
        that represents the data centre in its current resource state.
        '''
        req_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].requirements[resource] for resource in self.wrapped.features['node']]
        allocd_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].allocated[resource] for resource in self.wrapped.features['node']]

        remaining_vec = np.array(req_vec) - np.array(allocd_vec)

        lamda_func = lambda x: 1/x if x != 0. else 0.
        vec_func = np.vectorize(lamda_func)
        req_vec = vec_func(remaining_vec)
        mask_idx = tf.where(tf.squeeze(tf.equal(tf.reduce_sum(self.wrapped.dgl_graph.ndata['features'],-1),0.)))
        mask_idx = set(tf.squeeze(mask_idx).numpy())
        mask_idx.update(self.wrapped.curr_req_chosen)
        mask_idx = list(mask_idx)

        action_mask = np.ones(self.action_space.n)
        action_mask[mask_idx] = 0

        self.action_mask = action_mask
        
        self.wrapped.dgl_graph.ndata['orig_node'] = self.wrapped.dgl_graph.ndata['features']
        self.wrapped.dgl_graph.ndata['features'] =  tf.math.multiply(
                                                        self.wrapped.dgl_graph.ndata['features'],
                                                        req_vec
                                                    )
        
        self.wrapped.dgl_graph.ndata['h'] = self.wrapped.dgl_graph.ndata['features']#preprocessing.scale(
        self.action_assignments = tf.keras.activations.sigmoid(tf.math.subtract(self.wrapped.dgl_graph.ndata['features'],1))

        req_vec = [self.wrapped.manager.allocated_requests[self.wrapped.current_request].requirements[resource] for resource in self.wrapped.features['link']]

        remaining_vec = np.array(req_vec)

        lamda_func = lambda x: 1/x if x != 0. else 0.
        vec_func = np.vectorize(lamda_func)
        req_vec = vec_func(remaining_vec)

        self.wrapped.dgl_graph.edata['orig_edge'] = self.wrapped.dgl_graph.edata['features']
        self.wrapped.dgl_graph.edata['features'] =  tf.math.multiply(
                                                        self.wrapped.dgl_graph.edata['features'],
                                                        req_vec
                                                        )
        
        self.wrapped.dgl_graph.edata['h'] = self.wrapped.dgl_graph.edata['features']#preprocessing.scale(
        self.edge_features = tf.keras.activations.sigmoid(tf.math.subtract(self.wrapped.dgl_graph.edata['features'],1))

    def reset(self):

        obs = self.wrapped.reset()
        num_actions = self.wrapped.action_space.n
        self.chosen = np.zeros(num_actions)

        self.update_avail_actions()
        self.wrapped.new_req = False
        
        return {
            "avail_actions": self.action_assignments,
            "action_mask": self.action_mask,
            "network": obs,
            "edges":self.edge_features,
            "chosen_nodes":self.chosen,
            "edge_ids":tf.cast(self.wrapped.dgl_graph.edges()[0],tf.float32)
        }

    def step(self, action):
        '''
        Enact the step function from the wrapped environment, but return
        the more extensive (multi-faceted) observation so that the graph
        neural network component can deduce topological information about
        the data centre.

        Args:

        action (int): number that refers uniquely to a specific server in the data centre.
        '''
        orig_obs, rew, done, info = self.wrapped.step(action)
        
        if self.wrapped.new_req:
            self.chosen = np.zeros(self.wrapped.action_space.n)
            self.wrapped.new_req = False
        else:
            self.chosen[action] = 1

        self.update_avail_actions()

        obs = {
            "avail_actions": self.action_assignments,
            "action_mask": self.action_mask,
            "network": orig_obs,
            "edges":self.edge_features,
            "chosen_nodes":self.chosen,
            "edge_ids":tf.cast(self.wrapped.dgl_graph.edges()[0],tf.float32)
        } 

        return obs, rew, done, info

class ParametricActionsModel(TFModelV2):
    '''
    This class implements a policy architecture that is compatible with
    Rllib. It imports a graph neural network model implemented in DGL
    and uses this within the framework detailed in the RLlib documentation
    for custom policy models: 
        
        https://docs.ray.io/en/latest/rllib/rllib-concepts.html

    As such, arguments and return structure for all methods are the same 
    as detailed in this documentation and so this should be referenced
    for any general queries of how this is done.
    '''
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(OBS_SPACE_DIM, ),
                 action_embed_size=EMBEDDING_DIM,
                 **kw):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.save_embeddings = model_config['custom_model_config']['embedding_save_dir']
        self.config = model_config
        self.use_gnn = model_config['custom_model_config']['use_gnn']
        self.top_k = model_config['custom_model_config']['top_k']

        if self.use_gnn:
            logits_input_dim = model_config['custom_model_config']['obs_emb_dim'] + \
                                2*model_config['custom_model_config']['agg_dim']
        else:
            logits_input_dim = model_config['custom_model_config']['obs_emb_dim'] + \
                               len(self.config['custom_model_config']['features']['node'])

        self.action_logits_model = tf.keras.Sequential()
        self.action_logits_model.add(tf.keras.layers.Dense(1,input_shape=(None,logits_input_dim),name='logits_dense'))
        self.register_variables(self.action_logits_model.variables)

        if self.use_gnn:
            self.gnn = SAGE(agg_type=model_config['custom_model_config']['agg_type'],
                        agg_dim=model_config['custom_model_config']['agg_dim'],
                        agg_activation='relu',
                        num_mp_stages=model_config['custom_model_config']['num_mp_stages'])

            manager = NetworkManager(model_config['custom_model_config']['graph_dir'])
            self.dgl_graph = manager.network.to_dgl()

        #Define and register the observation embedding model
        final_obs_shape = (len(self.config['custom_model_config']['features']['node'])+1,)

        self.action_embed_model = FullyConnectedNetwork(
            Box(-1, 1, shape=final_obs_shape), 
                action_space, model_config['custom_model_config']['obs_emb_dim'],
                model_config, name + "_action_embed")

        self.register_variables(self.action_embed_model.variables())

        self.registered = False

    def forward(self, input_dict, state, seq_lens):

        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        obs = input_dict["obs"]["network"]
        edges = input_dict['obs']['edges']
        chosen_nodes = input_dict['obs']['chosen_nodes']

        new_feats = tf.concat([avail_actions,tf.expand_dims(chosen_nodes,-1)],-1)


        if self.use_gnn:
            if self.registered == False:
                tmp_feats = tf.ones([self.dgl_graph.number_of_nodes(),len(self.config['custom_model_config']['features']['node'])+1])
 
                self.dgl_graph.ndata['features'] = tmp_feats
                self.dgl_graph.ndata['h'] = self.dgl_graph.ndata['features']
                
                tmp_edge = tf.ones([self.dgl_graph.number_of_edges(),len(self.config['custom_model_config']['features']['link'])])
                self.dgl_graph.edata['features'] = tmp_edge
                self.dgl_graph.edata['h'] = self.dgl_graph.edata['features']
                
                self.gnn(self.dgl_graph,mode='no_sampling')
                print('gnn pass through')
                self.register_variables(self.gnn.variables)
                self.registered = True
            
            num_batches = avail_actions.shape[0]
            num_actions = avail_actions.shape[1]

            all_graphs = [self.dgl_graph] * num_batches
            all_graphs = dgl.batch(all_graphs)
            
            avail_actions = tf.reshape(new_feats,[num_batches*new_feats.shape[1],new_feats.shape[2]])
            edges = tf.reshape(edges,[num_batches*edges.shape[1],edges.shape[2]])
            
            all_graphs.ndata['features'] = avail_actions
            all_graphs.ndata['h'] = avail_actions
            
            all_graphs.edata['features'] = edges
            all_graphs.edata['h'] = edges
            embedded_actions = self.gnn(all_graphs,mode='no_sampling')
            embedded_actions = tf.reshape(embedded_actions,[num_batches,num_actions,embedded_actions.shape[-1]])

        else:
            embedded_actions = avail_actions

        if tf.math.reduce_sum(chosen_nodes,-1)[0].numpy() == 0:
            pooled_actions = tf.zeros((embedded_actions.shape[0],embedded_actions.shape[-1]))
        else:
            num_chosen = tf.reduce_sum(chosen_nodes,-1)
            batch = chosen_nodes.shape[0]
            num_actions = chosen_nodes.shape[1]
            emb_dim = self.config['custom_model_config']['agg_dim']

            chosen_nodes = tf.reshape(chosen_nodes,[batch*num_actions,1])
            chosen_nodes = tf.repeat(chosen_nodes,emb_dim,axis=-1)
            chosen_nodes = tf.expand_dims(chosen_nodes,0)
            chosen_nodes = tf.cast(tf.reshape(chosen_nodes,[batch,num_actions,emb_dim]),tf.float32)
            
            reduced_actions = tf.math.multiply(embedded_actions,chosen_nodes)
            pooled_actions = tf.reduce_sum(reduced_actions,-2)
            pooled_actions /= tf.cast(num_chosen.numpy()[0],tf.float32)

        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["network"]
        })

        action_embed = tf.concat([tf.cast(action_embed,tf.float32),tf.cast(pooled_actions,tf.float32)],axis=-1)
        intent_vector = tf.expand_dims(action_embed, 1)
        action_embed = tf.repeat(intent_vector,embedded_actions.shape[-2],axis=1)
        action_obs_embed = tf.concat([action_embed,tf.cast(embedded_actions,tf.float32)],-1)
        action_logits = tf.squeeze(self.action_logits_model(action_obs_embed),axis=-1)

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        logit_out = action_logits + tf.cast(inf_mask,tf.float32)

        if self.top_k is not None:
            action_mask = tf.zeros(action_mask.shape).numpy()
            values, indices = tf.math.top_k(logit_out,k=self.top_k)
            indices = indices.numpy()
            action_mask[0][indices] = 1
            inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
            logit_out = action_logits + tf.cast(inf_mask,tf.float32)

        if self.save_embeddings is not None:
            self.save(tf.squeeze(embedded_actions))

        return logit_out, state

    def value_function(self):
        return self.action_embed_model.value_function()
    
    def save(self,embeddings):
        np.savetxt(self.save_embeddings,embeddings.numpy())