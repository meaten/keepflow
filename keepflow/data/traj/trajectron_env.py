import itertools

from trajdata.data_structures import AgentType
from .build_traj_dataset import load_params

class EmptyEnv():
    def __init__(self, cfg):
        params = load_params(cfg)
        self.PredNodeType = [str(a).lstrip('AgentType.') for a in params['only_predict_list']]
        self.NodeType = [str(a).lstrip('AgentType.') for a in params['agent_list']]
        self.EdgeType = list(itertools.product(self.PredNodeType, self.NodeType))
        self.scenes = [EmptyScene(params['desired_dt'])]
    def get_edge_types(self):
        return self.EdgeType

    
class EmptyScene():
    def __init__(self, dt):
        self.dt = dt    
    
# def get_agent_edge_type(cfg):
#     if 'sim' in cfg.DATA.DATASET_NAME:
#         pred_agent_type_list = [AgentType.PEDESTRIAN]
#         agent_type_list = [AgentType.PEDESTRIAN]
#         edge_type_list = []
#     elif cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'jrdb']:
#         pred_agent_type_list = [AgentType.PEDESTRIAN]
#         agent_type_list = [AgentType.PEDESTRIAN]
#         edge_type_list = [(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)]
#     elif cfg.DATA.DATASET_NAME in ['sdd']:
#         pred_agent_type_list = [AgentType.PEDESTRIAN]
#         agent_type_list = [AgentType.PEDESTRIAN, AgentType.VEHICLE, AgentType.BICYCLE, AgentType.UNKNOWN]
#         # assume that few instances of other type of agents than pedestrians
#         edge_type_list = [(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN),
#                           (AgentType.PEDESTRIAN, AgentType.VEHICLE),
#                           (AgentType.PEDESTRIAN, AgentType.BICYCLE),
#                           (AgentType.PEDESTRIAN, AgentType.UNKNOWN)]
#     else:
#         raise ValueError(f'unknown dataset {cfg.DATA.DATASET_NAME}')
    
#     pred_agent_type_list = [str(a).lstrip('AgentType.') for a in pred_agent_type_list]
#     agent_type_list = [str(a).lstrip('AgentType.') for a in agent_type_list]
#     edge_type_list = [(str(a1).lstrip('AgentType.'), str(a2).lstrip('AgentType.')) for (a1, a2) in edge_type_list]
#     return pred_agent_type_list, agent_type_list, edge_type_list