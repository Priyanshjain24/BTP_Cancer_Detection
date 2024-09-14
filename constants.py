# Directories
D1_OG_DIR = '/home/priyansh/Downloads/datasets/PKG - Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1/Bone-Marrow-Cytomorphology/jpgs/BM_cytomorphology_data'
D1_DATA_DIR = '/home/priyansh/Downloads/datasets/d1_classify/cell_balanced_v2'
MODEL_DIR = '/home/priyansh/Downloads/code/weights/testing'
CHK_PTH = MODEL_DIR + '/last.pth'

# Categories
CANCER_CELLS = ["BLA", "FGC"]
NON_CANCER_CELLS = ["ART", "BAS", "EOS", "MMZ", "MON", "MYB", "NGB", "NGS", "OTH", "LYT", "NIF", "PLM"] + ["EBO", "PMO", "KSC", "LYI"]
CLASSES = ['cancer', 'non_cancer']
SPLITS = ['train', 'val']

# Parameters
SEED = 42
SPLIT_RATIO = [0.8, 0.2]
BALANCE = True
MODE = 'binary'
BATCH_SIZE = 16
NUM_EPOCHS = 1
LR = 0.0001
MOMENTUM = 0.9
CLASS_WEIGHTS = None
KMEANS = False
DEVICE = 'cuda'
PREDICTION_ONLY = False
MODEL = 'RESNET18'
DROPOUT = 0.2

# Testing Area
D2_TEST_INP_DIR = '/home/priyansh/Downloads/datasets/PKG - AML-Cytomorphology_MLL_Helmholtz_v1/data'
D2_TEST_OUT_DIR = '/home/priyansh/Downloads/datasets/seperate_testing/test_d2_output_v2'