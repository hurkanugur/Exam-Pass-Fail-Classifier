# -------------------------
# Paths
# -------------------------
DATA_PATH = "../data/student_exam_data.csv"
MODEL_PATH = "../model/exam_classifier.pth"

# -------------------------
# Training Hyperparameters
# -------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.005
NUM_EPOCHS = 1000
VAL_INTERVAL = 10

# -------------------------
# Dataset Splits
# -------------------------
SPLIT_DATASET = True  # Set False to use the full dataset for train/val/test
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = None  # Seed (integer) for reproducible dataset splits; set to None for fully random splits

# -------------------------
# Prediction / Classification
# -------------------------
CLASSIFICATION_THRESHOLD = 0.5  # Probability threshold to classify Pass vs Fail (used during inference/evaluation only)