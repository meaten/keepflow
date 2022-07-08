from yacs.config import CfgNode


def Build_VP_model(cfg: CfgNode, load: bool=False):
    if cfg.MODEL.TYPE == "SVG":
        from models.VP.svg import SVG
        model = SVG(cfg, load=load).cuda()
    if cfg.MODEL.TYPE == "PredRNN":
        if cfg.DATA.DATASET_NAME == "bair":
            from models.VP.predRNN import ActionCondPredRNN_v2
            model = ActionCondPredRNN_v2(cfg, load=load).cuda()
        else:
            from models.VP.predRNN import PredRNN_v2
            model = PredRNN_v2(cfg, load=load).cuda()
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model


