from yacs.config import CfgNode



def Build_TP_model(cfg: CfgNode, load: bool=False):
    if cfg.MODEL.TYPE == "socialGAN":
        from models.TP.socialGAN import socialGAN
        model = socialGAN(cfg, load=load).cuda()
    elif cfg.MODEL.TYPE == "socialLSTM":
        from models.TP.socialLSTM import socialLSTM
        model = socialLSTM(cfg, load=load).cuda()
    elif cfg.MODEL.TYPE == "ARFlow":
        from models.TP.TFCondARFlow import ARFlow
        model = ARFlow(cfg, load=load).cuda()
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
