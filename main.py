from src.models.model1 import trainer
from src.models.model1 import tester
from src.models.model1 import realtime_audio_predictor
from config import config


def main():

    # Select action
    inp = input("Train, Test or Predict [t/e/P] >>") # "e" # DEV
    if inp == "T" or inp == "t":
        if config.is_enable_overwrite_protection:
            exit_due_to_overwrite_protection_enabled()
        else:
            trainer.train()
    if inp == "E" or inp == "e":
        tester.test()
    if inp == "P" or inp == "p" or inp == "":
        realtime_audio_predictor.predict()


def exit_due_to_overwrite_protection_enabled():
    print(">>> ! OVERWRITE PROTECTION IS ENABLED -> EXITING ! <<<")
    exit(0)


if __name__ == "__main__":
    main()