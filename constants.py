MODEL = 'RESNET50'

# Directories
D1_OG_DIR = '/home/Drivehd2tb/garima/datasets/og_data/PKG - Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1/Bone-Marrow-Cytomorphology/jpgs/BM_cytomorphology_data'
D1_DATA_DIR = '/home/Drivehd2tb/garima/datasets/mod_data/d2_classify/sc/v1'
MODEL_DIR = f'/home/Drivehd2tb/garima/code/weights_new/d2_classify/{MODEL}/v1'
CHK_PTH = MODEL_DIR + '/last.pth'

# Categories
CANCER_CELLS = ["BLA", "FGC"]
NON_CANCER_CELLS = ["ART", "BAS", "EOS", "MMZ", "MON", "MYB", "NGB", "NGS", "OTH", "LYT", "NIF", "PLM"] + ["EBO", "PMO", "KSC", "LYI"]
CLASSES = ['cancer', 'non_cancer']
SPLITS = ['train', 'val']
MUTATIONS = ['CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']

# Parameters
SEED = 42
SPLIT_RATIO = [0.8, 0.2]
BALANCE = True
MODE = 'binary'
BATCH_SIZE = 128
NUM_EPOCHS = 1
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
CLASS_WEIGHTS = None
KMEANS = False
DEVICE = 'cuda'
PREDICTION_ONLY = False
MULTI_GPU = True

DROPOUT = 0.3
PATIENCE = 50
DELTA = 0.02
NUM_CLASSES = 4
FREEZE_LAYERS = False

# Testing Area
D2_TEST_INP_DIR = '/home/Drivehd2tb/garima/datasets/og_data/PKG - AML-Cytomorphology_MLL_Helmholtz_v1/data'
D2_TEST_OUT_DIR = '/home/Drivehd2tb/garima/datasets/mod_data/seperate/ft_v1/RESNET18/train'
D2_TEST_SC_DIR = '/home/Drivehd2tb/garima/datasets/mod_data/d2_classify/sc/v1'
D2_TEST_MIL_DIR = '/home/Drivehd2tb/garima/datasets/mod_data/d2_classify/mil/v1'
D2_TEST_PREDS_DIR = '/home/Drivehd2tb/garima/datasets/mod_data/d2_preds/v1'