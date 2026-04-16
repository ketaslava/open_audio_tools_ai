
# General
MODEL_NAME = "model1"
TRAIN_DATASET_PATH = "misc/datasets/dataset_dlmrvv_60hours/train"
TRAIN_DATASET_PORTION = 0.6
DEV_DATASET_PATH = "misc/datasets/dataset_dlmrvv_60hours/dev"
DEV_DATASET_PORTION = 0.5
TEST_DATASET_PATH = "misc/datasets/dataset_dlmrvv_60hours/test"
TEST_DATASET_PORTION = 1.0

# Data
SAMPLE_RATE = 48000

# Training
batch_size = 256
max_training_epochs = 22000
learning_rate = 0.0003
dropout = 0.5
training_log_step_epoch = 1
training_save_step_epoch = 1
