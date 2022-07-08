from yacs.config import CfgNode
from typing import Dict, List
from visualization.TP_visualizer import Visualizer


class MP_Visualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        
    def __call__(self, dict_list: List[Dict]) -> None:
        pass