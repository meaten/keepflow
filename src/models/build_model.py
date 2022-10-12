from unittest.loader import VALID_MODULE_NAME
from yacs.config import CfgNode
from typing import Dict
import torch
import torch.nn as nn
from copy import deepcopy
from models.TP.TP_models import Build_TP_model
from models.VP.VP_models import Build_VP_model
from models.MP.MP_models import Build_MP_model


def Build_Model(cfg: CfgNode) -> nn.Module:
    if cfg.MODEL.TYPE == "GT":
        return GT(cfg)
    elif cfg.MODEL.TYPE == "COPY_LAST":
        return COPY_LAST(cfg)

    if cfg.DATA.TASK == "TP":
        return Build_TP_model(cfg)
    elif cfg.DATA.TASK == "VP":
        return Build_VP_model(cfg)
    elif cfg.DATA.TASK == "MP":
        return Build_MP_model(cfg)
    else:
        raise(ValueError)

def build_loss(cfg: CfgNode):
    return lambda x: 0


class ModelTemplate(nn.Module):
    def __init__(self) -> None:
        super(ModelTemplate, self).__init__()

    def predict(self, data_dict):
        pass

    def update(self, data_dict):
        pass

class GT(ModelTemplate):
    def __init__(self, cfg: CfgNode) -> None:
        super(GT, self).__init__()

        self.loss_fn = build_loss(cfg)

    def predict(self, data_dict) -> Dict:
        data_dict["pred"] = deepcopy(data_dict["gt"])
        return data_dict

class COPY_LAST(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(COPY_LAST, self).__init__()
        self.pred_len = cfg.DATA.PREDICT_LENGTH
        self.loss = build_loss(cfg)

        self.task = cfg.DATA.TASK

    def predict(self, data_dict) -> Dict:
        size = data_dict["gt"].size()
        data_dict["pred"] = data_dict["obs"][-1].unsqueeze(0).expand(size)
        return data_dict



    