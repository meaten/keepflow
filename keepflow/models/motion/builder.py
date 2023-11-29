from yacs.config import CfgNode


def build_motion_pred_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "seq2seq":
        from .seq2seq import Seq2SeqModel
        model = Seq2SeqModel(cfg).cuda()
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
