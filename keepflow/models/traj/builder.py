from yacs.config import CfgNode


def build_traj_pred_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "GT_Dist":
        from .gtdist import GT_Dist
        model = GT_Dist(cfg).to(cfg.DEVICE)
        
    if cfg.MODEL.TYPE == "COPY_LAST":
        from .copy_last import COPY_LAST
        model = COPY_LAST(cfg).to(cfg.DEVICE)
        
    if cfg.MODEL.TYPE == 'socialLSTM':
        from .social_lstm import socialLSTM
        model = socialLSTM(cfg).to(cfg.DEVICE)
            
    elif cfg.MODEL.TYPE == "ARFlow":
        from .arflow import ARFlow
        model = ARFlow(cfg).to(cfg.DEVICE)
        
    elif cfg.MODEL.TYPE == "MID":
        from .mid import MID
        model = MID(cfg)
        model.to(cfg.DEVICE)
            
    elif cfg.MODEL.TYPE == "LED":
        from .led import LED
        model = LED(cfg).to(cfg.DEVICE)
        
    elif cfg.MODEL.TYPE == "Flomo":
        from .flomo import FloMo
        model = FloMo(cfg).to(cfg.DEVICE)
        
    elif cfg.MODEL.TYPE == "Trajectron":
        from .trajectron import Trajectron
        model = Trajectron(cfg).to(cfg.DEVICE)
        
    elif "FlowChain" in cfg.MODEL.TYPE:
        from .flowchain import FlowChainTraj
        model = FlowChainTraj(cfg).to(cfg.DEVICE)
    
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
        