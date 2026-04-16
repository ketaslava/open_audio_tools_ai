from src.models.model1.modules import dataset_processor
from src.models.model1 import config
import os
import numpy as np
from src.models.model1 import model
from src.models.model1.modules import utilities


def test():

    # Check dataset
    dataset_processor.check_data_in_dataset(config.TEST_DATASET_PATH)

    # Load test data
    print("== Loading Data ==")
    test_input_tensor, test_target_tensor = dataset_processor.get_tensor_for_model1_from_dataset(
        config.TEST_DATASET_PATH, config.TEST_DATASET_PORTION)

    print("== Data has loaded from Dataset ==")
    print(f"Test data size >> {len(test_input_tensor)}")

    # Folder with models to evaluate
    to_evaluation_dir = f"misc/models/model1/to_evaluation/"
    print("Searching for .pth files in:", to_evaluation_dir)

    pth_files = [
        f for f in os.listdir(to_evaluation_dir)
        if f.endswith(".pth")
    ]

    if not pth_files:
        print("ERROR: No model files found.")
        return

    print(f"Found {len(pth_files)} model weight files.")
    print()

    # Store errors for all files
    results = []

    # Loop all .pth model files
    for file_name in pth_files:
        print(f"Evaluating {file_name} ...")
        weights_path = os.path.join(to_evaluation_dir, file_name)

        # Predict
        predicted = model.predict(
            test_input_tensor,
            weights_path  # <-- add option in predict
        )

        # Normalize each prediction row
        norm_predicted = np.array([utilities.normalize_row(row) for row in predicted])
        print("Outputs:")
        print(norm_predicted)

        # Compute MAE error on normalized data
        abs_error = np.abs(norm_predicted - test_target_tensor)
        mean_error = abs_error.mean()

        print(f"  Mean error = {mean_error:.8f}")
        print()

        results.append((file_name, mean_error))

    # Sort results lowest → the highest error
    results.sort(key=lambda x: x[1])

    print("\n===== MODEL PERFORMANCE RATING =====")
    for rank, (file_name, err) in enumerate(results, start=1):
        print(f"{rank:2d}. {file_name:40s}  error={err:.8f}")
    print("====================================\n")
