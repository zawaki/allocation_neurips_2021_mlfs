from NetworkSimulator import Request
from AllocationFuncs.distribution_funcs import SingleResource

import random as rnd
import numpy as np

import sqlite3 as lite

import os

class SingleResourceRequest(Request):

    def __init__(self,config_file,id):

        super(SingleResourceRequest,self).__init__(config_file,id)
        self.node_capacity_quantity()

    def node_capacity_quantity(self):
        self.requirements['node'] = np.float32(rnd.randint(1,128))
        self.requirements['mem'] = np.float32(rnd.randint(1,128))
        self.requirements['ports'] = 1.0
        ht_norm = self.requirements['holding_time']
        self.requirements['holding_time'] = np.float32(rnd.randint(1,self.requirements['holding_time']))
        self.requirements['holding_time_feat'] = np.float32(self.requirements['holding_time']/ht_norm)