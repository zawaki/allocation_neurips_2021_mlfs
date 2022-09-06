from .Components import *
from .Requests import *
from .Network import *
from .utilz import Utilz as utl

import networkx as nx
import os
import pickle
from pathlib import Path
import copy

class NetworkManager(GlobalMessageBuffer):

    def __init__(self,root_directory):

        root_directory = Path(root_directory)
        root_directory = os.path.abspath(root_directory)

        if os.path.isfile('{}/custom_requests.py'.format(root_directory)):
            
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '{}'.format(root_directory))))
            import custom_requests
            custom_reqs = inspect.getmembers(custom_requests, inspect.isclass)
            
            for item in custom_reqs:
                globals()[item[0]] = item[1]
        
        self.root_directory = root_directory
        self.network = Network(self.root_directory)
        self.buffered_requests = {}
        self.allocated_requests = {}
        self.dropped_requests = {}
        self.request_count = {}

        self.node_pair_to_link = nx.get_edge_attributes(self.network.graph,'label')
        self.link_to_node_pair = {v:k for k,v in nx.get_edge_attributes(self.network.graph,'label').items()}

        self.removed_components = {}

    def util_var(self,resource):
        
        var = []
        util = 0
        comp_count = 0
        for component in self.network.components:
            try:
                utl = 1 - \
                        self.network.components[component].available[resource]/ \
                        self.network.components[component].maximum[resource]
                util += utl
                var.append(utl)
                comp_count += 1
            except:
                continue

        if comp_count != 0:
            util /= comp_count
            var = np.var(var)
        else:
            util = 0
            var = 0

        return util, var
    
    def packing_efficiency(self,sub_type):

        all_comps = self.network.components_of_sub_type(sub_type)

        used_comps = 0
        for comp in all_comps:
            if self.network.components[comp].allocations != {}:
                used_comps += 1

        return used_comps/len(all_comps)

        
    #externally used functions
    def get_request(self,request_sub_type):

        if request_sub_type not in self.request_count.keys():
            self.request_count[request_sub_type] = 0
    
        if request_sub_type in globals().keys():
            vm = globals()[request_sub_type]('{}/requests/{}.txt'.format(self.root_directory,request_sub_type),self.request_count[request_sub_type])
        else:
            vm = Request('config_{}.txt'.format(request_sub_type),self.request_count[request_sub_type])
        
        self._add_request(vm)
    
    def move_request_to_allocated(self,req_id):
        self.allocated_requests[req_id] = self.buffered_requests[req_id]
        del(self.buffered_requests[req_id])
    
    def move_request_to_buffered(self,req_id):
        self.buffered_requests[req_id] = self.allocated_requests[req_id]
        del(self.allocated_requests[req_id])
    
    def move_request_to_dropped(self,req_id):
        self.dropped_requests[req_id] = self.allocated_requests[req_id]
        del(self.allocated_requests[req_id])

    def allocate_and_remove(self,req_id,component_id,resource_id,resource_amount):
        self._allocate_resource_to_request(req_id,component_id,resource_id,resource_amount)
        if self.network.components[component_id].available[resource_id] == 0:
            self._remove_component(component_id)
    
    def allocate(self,req_id,component_id,resource_id,resource_amount):
        self._allocate_resource_to_request(req_id,component_id,resource_id,resource_amount)

    def de_allocate_and_replace(self,req_id):
        
        re_add = list(self.allocated_requests[req_id].allocations.keys())
        self._re_add_components(re_add)
        self._de_allocate_all_resource_for_request(req_id)
    
    def de_allocate(self,req_id):
        self._de_allocate_all_resource_for_request(req_id)

    def save(self,save_dir):

        filehandler = open(save_dir,'wb')
        pickle.dump(self,filehandler)
    
    def restore_custom_imports(self):

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '{}'.format(self.root_directory))))
        import custom_requests
        custom_reqs = inspect.getmembers(custom_requests, inspect.isclass)
        
        for item in custom_reqs:
            globals()[item[0]] = item[1]
        
        self.network.restore_custom_imports()

    #internally used functions
    def _re_add_adjecent_links(self,component_ids):

        node_ids = [component_id for component_id in component_ids if component_id in list(self.network.graph.nodes)]
        for node_id in node_ids:
        
            og_edges = [tuple(edge) for edge in self.graph_copy.edges(node_id)]
            current_edges = [tuple(edge) for edge in self.network.graph.edges(node_id)]

            edges_to_add = list(set(og_edges) - set(current_edges))
            edges = []
            for edge in edges_to_add:
                try:
                    edges.append(self.node_pair_to_link[(edge[0],edge[1])])
                except KeyError:
                    edges.append(self.node_pair_to_link[(edge[1],edge[0])])
            
            self._re_add_components(edges)

    def _re_add_components(self,component_ids):

        node_ids = [component_id for component_id in component_ids if component_id in list(self.graph_copy.nodes)]
        link_ids = [x for x in component_ids if x not in node_ids]

        for node_id in node_ids:
            try:
                self._re_add_component(node_id)
            except KeyError:
                continue

        for link_id in link_ids:
            try:
                self._re_add_component(link_id)
            except KeyError:
                continue

    def _remove_component(self,component_id):

        if component_id in list(self.graph_copy):
            self._remove_node(component_id)
        else:
            self._remove_link(component_id)

    def _re_add_component(self,component_id):

        if component_id in list(self.graph_copy):
            self._re_add_node(component_id)
        else:
            self._re_add_link(component_id)

    def _remove_node(self,node_id):

        #get ids of all edges adjecent to this node
        edges = [tuple(edge) for edge in self.network.graph.edges(node_id)]

        edge_ids = []
        for edge in edges:        
            try:
                edge_ids.append(self.node_pair_to_link[(edge[0],edge[1])])
            except KeyError:
                edge_ids.append(self.node_pair_to_link[(edge[1],edge[0])])
        
        #move node and adjecent edges from Network.components to NetworkManager.removed_components and remove from Network.graph
        for edge_id in edge_ids:
            self._remove_component_from_network_list(edge_id)

        self._remove_component_from_network_list(node_id)
        self.network.graph.remove_node(node_id)

    def _remove_link(self,link_id):
        #check that link has not already been removed due to another node removal
        try:
            self.network.graph.remove_edge(self.link_to_node_pair[link_id][0],self.link_to_node_pair[link_id][1])
        except:
            pass
        finally:
            #move edge from Network.components to NetworkManager.removed_components and remove from Network.graph
            self._remove_component_from_network_list(link_id)

    def _re_add_node(self,node_id):
        self._add_component_to_network_list(node_id)
        self.network.graph.add_node(node_id)
        self._re_add_adjecent_links([node_id])

    def _re_add_link(self,link_id):
        
        #check if src and dst nodes are currently in the graph, add edge between them if both are in
        src, dst = self.link_to_node_pair[link_id]
        if src in list(self.network.graph.nodes) and dst in list(self.network.graph.nodes):
            self.network.graph.add_edge(src,dst,label=link_id)
        
        #move link from NetworkManager.removed_components to Network.components
        self._add_component_to_network_list(link_id)
    
    def _remove_component_from_network_list(self,component_id):
        self.removed_components[component_id] = self.network.components[component_id]
        del(self.network.components[component_id])
    
    def _add_component_to_network_list(self,component_id):
        self.network.components[component_id] = self.removed_components[component_id]
        del(self.removed_components[component_id])
    
    def _add_request(self,vm):

        self.buffered_requests[vm.name] = vm
        self.request_count[vm.sub_type] += 1
    
    def _allocate_resource_to_request(self,request_id,component_id,resource_name,resource_amount=None):

        if resource_amount is None:
            resource_amount = self.network.components[component_id].available[resource_name]

        if self.network.components[component_id].available[resource_name] - resource_amount < 0.0:
            return False
        else:
            self.allocated_requests[request_id].allocate(component_id,resource_name,resource_amount)
            self.network.components[component_id].allocate(request_id,resource_name,resource_amount)
            return True

    def _de_allocate_resource_to_request(self,request_id,component_id,resource_name,resource_amount=None):

        if resource_amount is None:
            resource_amount = self.allocated_requests[request_id].allocations[component_id][resource_name]
        self.allocated_requests[request_id].de_allocate(component_id,resource_name,resource_amount)
        self.network.components[component_id].de_allocate(request_id,resource_name,resource_amount)

    def _de_allocate_all_resource_for_request(self,request_id):

        components = list(self.allocated_requests[request_id].allocations.keys())
        for component in components:
            resources = self.allocated_requests[request_id].allocations[component]
            for resource in list(resources.keys()):
                self._de_allocate_resource_to_request(request_id,component,resource)
    
    @staticmethod
    def update_graph_node_features(network_manager):
        
        for node in self.network.graph.nodes:
            self.network.graph.nodes[node]['features'] = self.network.component[node].feature