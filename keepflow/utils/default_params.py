from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 12345

_C.SAVE_DIR = "experiments"
_C.LOAD_TUNED = True

_C.DATA = CN()
# _C.DATA.PATH = "./keepflow/data/"
_C.DATA.PATH = "/home/maeda/home-local/keepflow/data/"
_C.DATA.TASK = "traj"
_C.DATA.DATASET_NAME = "zara1"
_C.DATA.OBSERVE_LENGTH = 8
_C.DATA.PREDICT_LENGTH = 12
_C.DATA.SKIP = 1

_C.DATA.TRAJ = CN()
_C.DATA.TRAJ.STATE = 'state_pva'  # should be fixed for trajectron encoder
_C.DATA.TRAJ.PRED_STATE = 'state_p'
_C.DATA.TRAJ.ACCEPT_NAN = False
_C.DATA.TRAJ.SCALE = 1.0
_C.DATA.TRAJ.USE_MAP = False
_C.DATA.AUGMENTATION = False

_C.MODEL = CN()
_C.MODEL.TYPE = "COPY_LAST"
_C.MODEL.TRAJ = CN()
_C.MODEL.TRAJ.ENCODER_TYPE = "trajectron"
_C.MODEL.FLOW = CN()
_C.MODEL.FLOW.ARCHITECTURE = 'realNVP'
_C.MODEL.FLOW.N_BLOCKS = 3
_C.MODEL.FLOW.N_HIDDEN = 2
_C.MODEL.FLOW.HIDDEN_SIZE = 16
_C.MODEL.FLOW.COND_SIZE = 16
_C.MODEL.FLOW.BASE_SIGMA = 0.1
_C.MODEL.FLOW.UPDATE_THRESHOLD = 1000.0
# Minimum Kish effective sample size, as a fraction of the particle count, for the
# fast update to be applied. Below it the cached particles no longer carry usable
# mass and the update would emit a density supported by a handful of them, so the
# previous forecast is kept instead. 0.0 disables the check.
_C.MODEL.FLOW.ESS_THRESHOLD = 0.0
_C.MODEL.FLOW.CNF_SOLVER = 'dopri5'
_C.MODEL.FLOW.NUM_STEP = None
_C.MODEL.FLOW.CNF_ATOL = 1e-2
_C.MODEL.FLOW.CNF_RTOL = 1e-2

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.OPTIMIZER= "adam"
_C.SOLVER.LR = 5e-3
_C.SOLVER.LR_MIN_RATIO = 1e-2
_C.SOLVER.ITER = 100
_C.SOLVER.EVAL_EVERY = 10
_C.SOLVER.USE_SCHEDULER = False
_C.SOLVER.EARLY_STOP = 3
_C.SOLVER.VALIDATION = True
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.DEQUANTIZE = False

_C.TEST = CN()
_C.TEST.N_TRIAL = 20
_C.TEST.KDE = False
# Trajectory samples drawn for density estimation (return_prob=True).
# For KDE-based baselines this is the sample budget fed to the KDE.
_C.TEST.KDE_N_SAMPLE = 10000

_C.USE_WANDB = False
_C.DEVICE = None
