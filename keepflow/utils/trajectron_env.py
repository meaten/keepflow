from trajdata.data_structures import AgentType

class EmptyEnv():
    def __init__(self, cfg):
        self.PredNodeType, self.NodeType, self.EdgeType = get_agent_edge_type(cfg)
        self.scenes = [EmptyScene(cfg)]
    def get_edge_types(self):
        return self.EdgeType

    
class EmptyScene():
    def __init__(self, cfg):
        self.dt = get_dt(cfg)

        
def get_dt(cfg):
    if 'sim' in cfg.DATA.DATASET_NAME:
        return 0.4
    elif cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        return 0.4
    elif cfg.DATA.DATASET_NAME in ['jrdb']:
        return 0.5
    else:
        raise ValueError(f'unknown dataset {cfg.DATA.DATASET_NAME}')
    
    
def get_agent_edge_type(cfg):
    if 'sim' in cfg.DATA.DATASET_NAME:
        pred_agent_type_list = [AgentType.PEDESTRIAN]
        agent_type_list = [AgentType.PEDESTRIAN]
        edge_type_list = []
    elif cfg.DATA.DATASET_NAME in ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'jrdb']:
        pred_agent_type_list = [AgentType.PEDESTRIAN]
        agent_type_list = [AgentType.PEDESTRIAN]
        edge_type_list = [(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)]
    else:
        raise ValueError(f'unknown dataset {cfg.DATA.DATASET_NAME}')
    
    pred_agent_type_list = [str(a).lstrip('AgentType.') for a in pred_agent_type_list]
    agent_type_list = [str(a).lstrip('AgentType.') for a in agent_type_list]
    edge_type_list = [(str(a1).lstrip('AgentType.'), str(a2).lstrip('AgentType.')) for (a1, a2) in edge_type_list]
    return pred_agent_type_list, agent_type_list, edge_type_list