from yacs.config import CfgNode


def Build_TP_model(cfg: CfgNode):
    if cfg.MODEL.TYPE == "socialGAN":
        from models.TP.socialGAN import socialGAN
        model = socialGAN(cfg).cuda()
        
    elif cfg.MODEL.TYPE == "socialLSTM":
        from models.TP.socialLSTM import socialLSTM
        model = socialLSTM(cfg).cuda()
        
    elif cfg.MODEL.TYPE == "ARFlow":
        from models.TP.TFCondARFlow import ARFlow
        model = ARFlow(cfg).cuda()
        
    elif "fastpredNF" in cfg.MODEL.TYPE:
        from models.TP.fastpredNF import fastpredNF_TP
        model = fastpredNF_TP(cfg).cuda()
        
    else:
        raise(ValueError, f"unknown model type: {cfg.MODEL.TYPE}")

    return model
