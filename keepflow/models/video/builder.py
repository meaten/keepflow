from yacs.config import CfgNode


def build_video_pred_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "SVG":
        from .svg import SVG
        model = SVG(cfg).cuda()
    if cfg.MODEL.TYPE == "PredRNN":
        if cfg.DATA.DATASET_NAME == "bair":
            from .predrnn import ActionCondPredRNN_v2
            model = ActionCondPredRNN_v2(cfg).cuda()
        else:
            from .predrnn import PredRNN_v2
            model = PredRNN_v2(cfg).cuda()
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model


