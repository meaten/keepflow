from yacs.config import CfgNode


def build_traj_pred_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "COPY_LAST":
        from .copy_last import COPY_LAST
        model = COPY_LAST(cfg).to(cfg.DEVICE)
        
    elif cfg.MODEL.TYPE == 'socialLSTM':
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
        
    elif cfg.MODEL.TYPE == "SingularTrajectory":
        from .sintraj_wo_aa import SingularTrajectory_wo_AA
        model = SingularTrajectory_wo_AA(cfg).to(cfg.DEVICE)
        
    elif "FlowChainProcess" in cfg.MODEL.TYPE:
        from .flowchainprocess import ContinuousFlowChainProcessTraj
        model = ContinuousFlowChainProcessTraj(cfg).to(cfg.DEVICE)
        
    elif "ContinuousFlowChainOnestep" in cfg.MODEL.TYPE:
        from .continuousflowchain_onestep import ContinuousFlowChainTraj
        model = ContinuousFlowChainTraj(cfg).to(cfg.DEVICE)
        
    elif "ContinuousFlowChain" in cfg.MODEL.TYPE:
        from .continuousflowchain import ContinuousFlowChainTraj
        model = ContinuousFlowChainTraj(cfg).to(cfg.DEVICE)

    elif "FlowMatchingChainOnestep" in cfg.MODEL.TYPE:
        if cfg.MODEL.TRAJ.ENCODER_TYPE == "trajectron":
            from .flowmatchingchain_onestep import FlowMatchingChainTraj
            model = FlowMatchingChainTraj(cfg).to(cfg.DEVICE)
        elif cfg.MODEL.TRAJ.ENCODER_TYPE == "social_transformer":
            from .flowmatchingchain_onestep import FlowMatchingChainTraj_transformer
            model = FlowMatchingChainTraj_transformer(cfg).to(cfg.DEVICE)
    elif "FlowMatchingChain" in cfg.MODEL.TYPE:
        from .flowmatchingchain import FlowMatchingChainTraj
        model = FlowMatchingChainTraj(cfg).to(cfg.DEVICE)
        
    elif "FlowChain" in cfg.MODEL.TYPE:
        from .flowchain import FlowChainTraj
        model = FlowChainTraj(cfg).to(cfg.DEVICE)
    
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
        