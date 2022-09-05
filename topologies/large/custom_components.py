from NetworkSimulator import Component
from AllocationFuncs.distribution_funcs import SingleResource
import numpy as np
import random as rnd

"""These classes are currently hard-coded to be a single node resource (no resource at switches) + single link resource
set of components
"""

class SRLink(Component):

    def __init__(self,config_file,id):

        super(SRLink,self).__init__(config_file,id)
        self.init_link_resource()

    def init_link_resource(self):
        self.available['ports'] = np.float32(self.available['ports'])

class RALink(Component):
    
    def __init__(self,config_file,id):

        super(RALink,self).__init__(config_file,id)
        self.init_link_resource()

    def init_link_resource(self):
        self.available['ports'] = np.float32(self.available['ports'])
        
class ACLink(Component):
    
    def __init__(self,config_file,id):

        super(ACLink,self).__init__(config_file,id)
        self.init_link_resource()

    def init_link_resource(self):
        self.available['ports'] = np.float32(self.available['ports'])

class Aggregate(Component):

    def __init__(self,config_file,id):

        super(Aggregate,self).__init__(config_file,id)
        self.init_node_resource()

    def init_node_resource(self):
        self.available['node'] = np.float32(self.available['node'])
        self.available['mem'] = np.float32(self.available['mem'])


class Core(Component):

    def __init__(self,config_file,id):

        super(Core,self).__init__(config_file,id)
        self.init_node_resource()

    def init_node_resource(self):
        self.available['node'] = np.float32(self.available['node'])
        self.available['mem'] = np.float32(self.available['mem'])

class Rack(Component):

    def __init__(self,config_file,id):

        super(Rack,self).__init__(config_file,id)
        self.init_node_resource()

    def init_node_resource(self):
        self.available['node'] = np.float32(self.available['node'])
        self.available['mem'] = np.float32(self.available['mem'])


class Resource(Component):

    def __init__(self,config_file,id):

        super(Resource,self).__init__(config_file,id)
        self.init_node_resource()

    def init_node_resource(self):
        self.available['node'] = np.float32(self.available['node'])
        self.available['mem'] = np.float32(self.available['mem'])