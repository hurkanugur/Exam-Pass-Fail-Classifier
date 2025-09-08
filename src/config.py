# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/exam_classifier.pth"
NORM_PARAMS_PATH = "../model/norm_params.pkl"
DATASET_CSV_PATH = "../data/student_exam_data.csv"

# -------------------------
# Training Hyperparameters
# -------------------------
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 64
NUM_EPOCHS = 1000
VAL_INTERVAL = 1

# -------------------------
# Dataset Splits
# -------------------------
SPLIT_DATASET = True
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
SPLIT_RANDOMIZATION_SEED = 42   # Int -> Reproducible splits | None -> Fully random splits

# -------------------------
# Prediction / Classification
# -------------------------
CLASSIFICATION_THRESHOLD = 0.5