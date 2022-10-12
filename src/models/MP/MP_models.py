from yacs.config import CfgNode


def Build_MP_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "seq2seq":
        from models.MP.seq2seq import Seq2SeqModel
        model = Seq2SeqModel(cfg).cuda()
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
