import numpy as np
import json
import os
from networkx.readwrite import json_graph
from pathlib import Path

class Utilz:

    @staticmethod
    def load_json(config_file):

        config_file = Path(config_file)
        config_file = os.path.abspath(config_file)

        assert os.path.isfile(config_file), 'Invalid configuration file path.'
        with open(config_file) as json_file:
            d = json.load(json_file)
        
        json_file.close()
        return d
        #     return json.load(json_file)
        # json_file.close()
    
    @staticmethod
    def save_graph_as_json(graph,save_file):
        data = json_graph.node_link_data(graph)
        with open(save_file,'w+') as f:
            json.dump(data,f)
    
    @staticmethod
    def save_data_as_json(item, save_file):
        with open(save_file, 'w+') as f:
            json.dump(item,f)