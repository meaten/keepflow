from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 12345

_C.OUTPUT_DIR = "output"
_C.LOAD_TUNED = False

_C.DATA = CN()
_C.DATA.PATH = "/home-local/maeda/fastpredNF/data/"
_C.DATA.TASK = "TP"
_C.DATA.DATASET_NAME = "zara1"
_C.DATA.OBSERVE_LENGTH = 8
_C.DATA.PREDICT_LENGTH = 8
_C.DATA.SKIP = 1
_C.DATA.BATCH_SIZE = 64
_C.DATA.BATCH_SIZE_TEST = 1
_C.DATA.NUM_WORKERS = 4


_C.MODEL = CN()
_C.MODEL.TYPE = "COPY_LAST"
#_C.MODEL.TYPE = "GT"
#_C.MODEL.TYPE = "socialGAN"

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER= "adam"
_C.SOLVER.ITER = 500
_C.SOLVER.SAVE_EVERY = 20
_C.SOLVER.TRY_LOAD = True
_C.SOLVER.USE_SCHEDULER = False
_C.SOLVER.VALIDATION = True

_C.TEST = CN()
_C.TEST.N_TRIAL = 20

_C.USE_WANDB = False
