from src.models.model1 import model
from src.models.model1.modules import dataset_processor
from src.models.model1 import config


def train():

    # Check dataset
    dataset_processor.check_data_in_dataset(config.TRAIN_DATASET_PATH)
    dataset_processor.check_data_in_dataset(config.DEV_DATASET_PATH)

    # Get data
    print("== Loading Data ==")
    train_input_tensor, train_target_tensor = dataset_processor.get_tensor_for_model1_from_dataset(
        config.TRAIN_DATASET_PATH, config.TRAIN_DATASET_PORTION)
    dev_input_tensor, dev_target_tensor = dataset_processor.get_tensor_for_model1_from_dataset(
        config.DEV_DATASET_PATH, config.DEV_DATASET_PORTION)
    print("== Data has loaded from Dataset ==")
    print(f"Train data size >> {len(train_input_tensor)}")

    # Train
    print("== Training ==")
    model.train(train_input_tensor, train_target_tensor, dev_input_tensor, dev_target_tensor,
                "misc/models/model1/model.pth")
    print("== Training has finished Successfully ==")
