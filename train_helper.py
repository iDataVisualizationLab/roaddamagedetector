from detectron2 import model_zoo
from detectron2.config import get_cfg


def configure_model(OUTPUT_DIR, base_config_file, base_weight_file=None, max_iter=20000, num_gpus=2, ims_per_batch=16,
                    learning_rate=0.00025, sizes=[32, 64, 128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0]):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = OUTPUT_DIR

    cfg.merge_from_file(model_zoo.get_config_file(base_config_file))

    cfg.DATASETS.TRAIN = ("road_damage_train",)
    cfg.DATASETS.TEST = ()
    # # for validation
    cfg.DATASETS.TEST = ("road_damage_eval",)
    cfg.TEST.EVAL_PERIOD = 5000

    cfg.DATALOADER.NUM_WORKERS = ims_per_batch

    if base_weight_file is not None:
        cfg.MODEL.WEIGHTS = base_weight_file

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.REFERENCE_WORLD_SIZE = num_gpus

    cfg.SOLVER.MAX_ITER = max_iter

    cfg.SOLVER.BASE_LR = learning_rate

    cfg.SOLVER.MOMENTUM = 0.9

    cfg.SOLVER.NESTEROV = False

    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = (30000,)

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"

    # Save a checkpoint after every this number of iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 12500 # 4096   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [sizes]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [aspect_ratios]
    return cfg



