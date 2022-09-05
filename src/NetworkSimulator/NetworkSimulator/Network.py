from .GlobalMessageBuffer import GlobalMessageBuffer
from .Components import *
from .utilz import Utilz as utl

import networkx as nx
import json
import os
import pickle
import sys
import matplotlib.pyplot as plt
import inspect
import numpy as np

import os
os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
import dgl
from collections import OrderedDict
import tensorflow as tf

class Network(GlobalMessageBuffer):

    def __init__(self,root_directory):

        #import custom components if they exist
        if os.path.isfile('{}/custom_components.py'.format(root_directory)):

            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '{}'.format(root_directory))))
            import custom_components
            custom_comps = inspect.getmembers(custom_components, inspect.isclass)
            
            for item in custom_comps:
                globals()[item[0]] = item[1]
            
        self.root_directory = root_directory

        #load graph defined in config directory
        loaded_graph = utl.load_json('{}/graph.txt'.format(self.root_directory))
        self.graph = nx.readwrite.node_link_graph(loaded_graph)

        try:
            self.pos = nx.get_node_attributes(self.graph,'pos')
        except:
            pass

        #check that a config file (JSON) exists for all components specified in graph
        node_labels = list(self.graph.nodes)
        link_labels = []
        for edge in self.graph.edges:
            link_labels.append(self.graph[edge[0]][edge[1]]['label'])
        component_labels = node_labels + link_labels
        self._check_config_and_class_exist(component_labels)

        #create component dict
        self.components = {}
        for component in component_labels:
            component_type = component.split('_')[0]
            component_id = int(component.split('_')[1])
            config_component = '{}/components/{}.txt'.format(self.root_directory,component_type)
            if component_type in globals().keys():
                new_component = globals()[component_type](config_component,component_id)
            else:
                new_component = Component(config_component,component_id)
            self.components['{}_{}'.format(component_type,component_id)] = new_component

    #externally used functions
    def link_connecting_nodes(self,node1,node2,label='label'):

        try:
            link_label = self.graph[node1][node2][label]

        except KeyError:
            link_label = None
        
        return link_label
    
    def links_in_path(self,path):

        links = []

        for i in range(len(path)-1):
            links.append(self.link_connecting_nodes(path[i],path[i+1]))
        
        return links
    
    def core_components_in_path(self,path):

        return self.all_components_in_path(path)[1:-1]

    def all_components_in_path(self,path):

        links = self.links_in_path(path)
        nodes = path

        components = links + nodes
        components[1::2] = links
        components[::2] = nodes

        return components

    def components_of_type(self,component_type):

        component_type_sub_list = list(dict((key, value) for key, value in self.components.items() \
                                                if value.type == component_type).keys())

        return component_type_sub_list

    def components_of_sub_type(self,component_type):

        component_type_sub_list = list(dict((key, value) for key, value in self.components.items() \
                                                if value.sub_type == component_type).keys())
        
        return component_type_sub_list
    
    def components_with_resource(self,resource_type):

        resource_type_sub_list = list(dict((key, value) for key, value in self.components.items() \
                                                if resource_type in value.available.keys()))
        
        return resource_type_sub_list

    def show_graph(self,with_labels=False,size=None):
        if size is not None:
            plt.figure(figsize=size)
        try:
            nx.draw(self.graph,self.pos,with_labels=with_labels)
            plt.show()
        except:
            nx.draw(self.graph,with_labels=with_labels)
            plt.show()

    def line_conversion(self):

        for edge in self.graph.edges():
            label = self.graph.edges[edge]['label']
            self.components[label].available['src'] = edge[0]
            self.components[label].available['dst'] = edge[1]

        #convert to line-graph and re-name edges with original edge names
        line = nx.line_graph(self.graph)
        line_node_conversion_map = {node:self.graph.edges[node]['label'] for node in line.nodes()}
        line = nx.relabel_nodes(line,line_node_conversion_map)
        
        self.graph = line

    def reverse_line_conversion(self):
        primal_graph = nx.Graph()
        
        for node in self.graph.nodes():
            primal_graph.add_edge(self.components[node].available['src'],self.components[node].available['dst'])
        
        self.graph = primal_graph

    def combi_to_dgl(self,*argv):
        '''Return a DGLGraph version of the 
        self.graph networkx graph. List the names 
        of the attributes to be carried over to features
        with argv. Scalar and list based attributes will be appended
        in the order specified by argv to create the final feature 
        for each node.
        '''
        tf.config.experimental_run_functions_eagerly(True)
        names = np.sort(self.graph.nodes())
        n_ids = np.arange(len(names))
        netx = self._netx_with_attrs(*argv)
        g = dgl.DGLGraph()

        if argv != ():
            g.from_networkx(netx,node_attrs=['features'])
        else:
            g.from_networkx(netx)

        g.ndata['_ID'] = n_ids
        g.ndata['_name'] = names
        
        return g
    
    def to_dgl(self,*argv):
        '''Return a DGLGraph version of the 
        self.graph networkx graph. List the names 
        of the attributes to be carried over to features
        with argv. Scalar and list based attributes will be appended
        in the order specified by argv to create the final feature 
        for each node.
        '''
        tf.config.experimental_run_functions_eagerly(True)
        names = np.sort(self.graph.nodes())
        n_ids = np.arange(len(names))
        netx = self._netx_with_attrs(*argv)
        g = dgl.DGLGraph()

        if argv != ():
            g.from_networkx(netx,node_attrs=['features'])
        else:
            g.from_networkx(netx)

        g.ndata['_ID'] = n_ids
        g.ndata['_name'] = names
        g.ndata['_pos'] = np.array(
                                list(
                                dict(
                                OrderedDict(
                                sorted(
                                self.pos.items()
                                ))).values()))
        
        return g

    def _netx_with_attrs(self,*argv):            

        graph = self.graph

        for node in self.graph.nodes():
            graph.nodes[node]['features'] = []
            for k in argv:
                v = self.components[node].available[k]
                graph.nodes[node]['features'] += v if isinstance(v,list) else [v]

        return graph

    def to_dgl_with_edges(self,features):
        '''Return a DGLGraph version of the 
        self.graph networkx graph. List the names 
        of the attributes to be carried over to features
        with argv. Scalar and list based attributes will be appended
        in the order specified by argv to create the final feature 
        for each node.
        '''
        tf.config.experimental_run_functions_eagerly(True)
        n_names = np.sort(self.graph.nodes())
        n_ids = np.arange(len(n_names))

        netx = self._netx_with_all_attrs(features)

        g = dgl.DGLGraph()

        g.from_networkx(netx,node_attrs=['features'],edge_attrs=['features'])#,'_name'])

        g.ndata['_ID'] = n_ids
        g.ndata['_name'] = n_names
        g.ndata['_pos'] = np.array(
                                list(
                                dict(
                                OrderedDict(
                                sorted(
                                self.pos.items()
                                ))).values()))

        return g

    def _netx_with_all_attrs(self,features):            

        graph = self.graph

        nodes = graph.nodes()
        node_attrs = {name:{'features':[self.components[name].available[x] for x in features['node']]} for name in nodes}

        edges = graph.edges()
        edge_attrs = {name:{'features':[self.components[graph.edges[name]['label']].available[x] for x in features['link']]} for name in edges}

        nx.set_node_attributes(graph,node_attrs)
        nx.set_edge_attributes(graph,edge_attrs)
        # print('set attributes')
        return graph

    def netx_lb_routing_weights(self):            
        
        graph = self.graph
        f = 1.0

        for edge in self.graph.edges():
            graph.edges[edge]['weight'] = 0
            edge_id = graph.edges[edge]['label']
            graph.edges[edge]['_name'] = edge

            v = f*(self.components[edge_id].available['ports'])
            graph.edges[edge]['weight'] += v

            v = (1-f)*self.components[edge_id].available['length']
            graph.edges[edge]['weight'] += v

        return graph

    def save_graph_image(self,save_dir):
        
        try:
            nx.draw(self.graph,self.pos)
            plt.savefig(save_dir)
        except:
            nx.draw(self.graph,with_labels=True)
            plt.savefig(save_dir)

    def restore_custom_imports(self):

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '{}'.format(self.root_directory))))
        import custom_components
        custom_comps = inspect.getmembers(custom_components, inspect.isclass)
        
        for item in custom_comps:
            globals()[item[0]] = item[1]
    
    #internally used functions
    def _check_config_and_class_exist(self,component_list):

        if not isinstance(component_list,list):
            component_list = [component_list]
        
        component_types = set([label.split('_',1)[0] for label in component_list])

        for component in component_types:
            assert os.path.isfile('{}/components/{}.txt'.format(self.root_directory, component)), \
                    '{} component specified in graph configuration but no "config_{}.txt" file found.'.format(component,component)