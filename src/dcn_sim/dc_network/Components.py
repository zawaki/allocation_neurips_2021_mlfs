from .GlobalMessageBuffer import GlobalMessageBuffer
from .utilz import Utilz as utl

class Component(GlobalMessageBuffer):
    """Non-specific component class which can be initialized using an appropriately made JSON file.
    See README.md for details
    """
    def __init__(self,config_file,comp_id):

        config = utl.load_json(config_file)
      
        self.available = {}
        self.maximum = {}

        for key in config.keys():
            if key == 'type':
                self.type = config[key]
            elif key == 'sub-type':
                self.sub_type = config[key]
            else:
                self.available[key] = config[key]
                self.maximum[key] = config[key]

        self.id = comp_id
        self.allocations = {}

        self.name = '{}_{}'.format(self.sub_type,self.id)

    def allocate(self,request_id,resource_name,resource_amount):

        self.available[resource_name] -= resource_amount

        if request_id not in self.allocations.keys():
            self.allocations[request_id] = {}
        if resource_name not in self.allocations[request_id].keys():
            self.allocations[request_id][resource_name] = resource_amount
        else:
            self.allocations[request_id][resource_name] += resource_amount

    def de_allocate(self,request_id,resource_name,resource_amount):

        self.available[resource_name] += resource_amount
        self.allocations[request_id][resource_name] -= resource_amount
        if self.allocations[request_id][resource_name] == 0.0:
            del(self.allocations[request_id][resource_name])
            if not self.allocations[request_id]:
                del(self.allocations[request_id])

    def __str__(self):
        return self.name