import tensorflow as tf
from .aggregators import *
import json

class SAGE(tf.keras.Model):
    
    def __init__(self,
                agg_type=None,
                ro_type=None,
                ro_dim=None,
                ro_activation=None,
                agg_dim=None,
                agg_activation=None,
                num_mp_stages=None):
        '''
        Args:

        agg_type (str): which aggregator layer to use, referenced by name and their
                            implementation in aggregators.py. 'MaxPool' and
                            'MeanPool2' currently supported.
        ro_type (str): which readout layer to use, referenced by name and their
                            implementation in aggregators.py. 'SingleLayerReadout'
                            currently supported.
        ro_dim (int): how many units in the readout layer (assumes single-layer).
        ro_activation (str): what activation to use in the readout layer (value of argument 
                                corresponds to those used in Tensorflow).
        agg_dim (int): how many units in the aggreagor layer (assumes single-layer neural network
                        per aggregator layer component).
        agg_acviations (str): what activation to use in the readout layer (value of argument 
                                corresponds to those used in Tensorflow).
        num_mp_stages (int): how many message-passing stages will there be (i.e. how many aggregation
                                layers should be generated).
        '''
        super(SAGE,self).__init__()

        self.model_params = {
                            'agg_type':agg_type,
                            'ro_type':ro_type,
                            'ro_dim':ro_dim,
                            'ro_activation':ro_activation,
                            'agg_dim':agg_dim,
                            'agg_activation':agg_activation,
                            'num_mp_stages':num_mp_stages
                            }

        self.num_mp_stages = num_mp_stages
        self.use_readout = ro_type is not None
        self.aggregators = []

        for _ in range(num_mp_stages):
            if agg_type == 'MaxPool':
                self.aggregators.append(MaxPool(agg_dim,activation=agg_activation))
            if agg_type == 'MeanPool2':
                self.aggregators.append(MeanPool2(agg_dim,activation=agg_activation))
        if self.use_readout:
            if ro_type == 'SingleLayerReadout':
                self.readout = SingleLayerReadout(ro_dim,activation=ro_activation)


    def call(self,graph,mode='sampling'):
        '''

        '''
        if mode == 'sampling':
            
            hidden = graph.layers[-1].data['features']
            graph.layers[-1].data['h'] = graph.layers[-1].data['features']
            for layer, aggregator in enumerate(self.aggregators):
                hidden = aggregator(graph,hidden,stage=layer,mp_stages=self.num_mp_stages,mode=mode)
                
            if self.use_readout:
                return self.readout(tf.math.l2_normalize(graph.layers[-1].data['h'],-1))
            else:
                return tf.math.l2_normalize(graph.layers[-1].data['h'],-1)

        if mode == 'no_sampling':

            hidden = graph.ndata['features']
            graph.ndata['h'] = graph.ndata['features']

            for aggregator in self.aggregators:
                hidden = aggregator(graph,hidden,mode=mode)

            if self.use_readout:
                return self.readout(tf.math.l2_normalize(graph.ndata['h'],-1))
            else:
                return tf.math.l2_normalize(graph.ndata['h'],-1)
    
    @staticmethod
    def load_model(weights_dir,weights_file):

        with open('{}/model_params.txt'.format(weights_dir),'r') as f:
            model_params = json.load(f)

        model = SAGE(**model_params)
        model.load_weights('{}/{}'.format(weights_dir,weights_file))

        return model