from .GlobalMessageBuffer import GlobalMessageBuffer
from .utilz import Utilz as utl

class Request(GlobalMessageBuffer):

    def __init__(self,config_file,req_id):

        config = utl.load_json(config_file)
        
        self.requirements = {}
        self.allocated = {}
        
        for key in config.keys():
            if key == 'type':
                self.type = config[key]
            elif key == 'sub-type':
                self.sub_type = config[key]
            else:
                self.requirements[key] = config[key]
                self.allocated[key] = 0

        self.id = req_id
        self.allocations = {}
        self.path_for_pair = {}
        self.pairs_with_link = {}

        self.name = '{}_{}'.format(self.sub_type,self.id)

    def allocate(self,component_id,resource_name,resource_amount):

        self.allocated[resource_name] += resource_amount

        if component_id not in self.allocations.keys():
            self.allocations[component_id] = {}
        if resource_name not in self.allocations[component_id].keys():
            self.allocations[component_id][resource_name] = resource_amount
        else:
            self.allocations[component_id][resource_name] += resource_amount

    def de_allocate(self,component_id,resource_name,resource_amount):

        if not self.allocations[component_id][resource_name] - resource_amount >= 0.0:
            return False

        self.allocations[component_id][resource_name] -= resource_amount

        if self.allocations[component_id][resource_name] == 0.0:
            del(self.allocations[component_id][resource_name])
            if not self.allocations[component_id]:
                del(self.allocations[component_id])
        
        return True
        
    def __str__(self):
        return self.name