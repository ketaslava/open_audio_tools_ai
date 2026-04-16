import statistics
import torch


def evaluate_prediction_model1(tensor_predict, tensor_target):
    # Full data evaluation
    evaluate_prediction_accuracy(
        tensor_predict, tensor_target, 6)
    evaluate_prediction_accuracy_score(
        tensor_predict, tensor_target, 14)


def evaluate_prediction_accuracy(tensor_predict, tensor_target, count):
    print("* Prediction raw accuracy :")
    i = 0
    for predict in tensor_predict:
        print("pred: " + str(float(tensor_target[i][0]))[:6] + "  loss: " +
              str(float(tensor_target[i][0]) - float(predict[0]))[:6])
        i += 1
        if i == count:
            return


def evaluate_prediction_accuracy_score(tensor_predict, tensor_target, count):
    print("* Prediction score accuracy:")
    score_delta_array = []
    i = 0
    for predict in tensor_predict:
        # Calc
        #score_pred = calc_nn_data.data2score(predict[0])
        #score_real = calc_nn_data.data2score(tensor_target[i][0])
        #score_delta = (score_pred[0] + score_pred[1]) - \
        #              (score_real[0] + score_real[1])
        # Print
        #if i < count:
        #    print("pred: " + str(score_pred) + "  real: " +
        #          str(score_real) + "  delta: " + str(score_delta))
        # Save data
        #score_delta_array.append(score_delta)
        # Loop
        i += 1
    # Delta stats ( average, abs average, median )
    delta_average = 0
    delta_abs_average = 0
    for val in score_delta_array:
        delta_average += val
        delta_abs_average += abs(val)
    delta_average /= len(score_delta_array)
    delta_abs_average /= len(score_delta_array)
    delta_median = statistics.median(score_delta_array)
    # Print
    print("Average delta: " + str(delta_average))
    print("Average ABS delta: " + str(delta_abs_average))
    print("Median delta: " + str(delta_median))


def evaluate_prediction_model2(tensor_input,
                               tensor_model1_predict,
                               tensor_model1_target,
                               tensor_model2_predict,
                               tensor_model2_target):
    # Count
    count = 20

    # Convert data type
    if type(tensor_input) is torch.Tensor:
        tensor_input = tensor_input.tolist()
    if type(tensor_model1_predict) is torch.Tensor:
        tensor_model1_predict = tensor_model1_predict.tolist()
    if type(tensor_model1_target) is torch.Tensor:
        tensor_model1_target = tensor_model1_target.tolist()
    if type(tensor_model2_predict) is torch.Tensor:
        tensor_model2_predict = tensor_model2_predict.tolist()
    if type(tensor_model2_target) is torch.Tensor:
        tensor_model2_target = tensor_model2_target.tolist()

    # Process
    score_delta_array = []
    good_pred_score_delta_array = []
    i = 0
    while i < len(tensor_input):
        # Get data
        block_input = tensor_input[i]
        val_model1_predict = tensor_model1_predict[i][0]
        val_model1_target = tensor_model1_target[i][0]
        val_model2_predict = tensor_model2_predict[i][0]
        val_model2_target = tensor_model2_target[i][0]

        is_predict_good = True
        if val_model2_predict < 0.5:
            is_predict_good = False

        # Calc score accuracy
        #score_pred = calc_nn_data.data2score(val_model1_predict)
        #score_real = calc_nn_data.data2score(val_model1_target)
        #score_delta = (score_pred[0] + score_pred[1]) - \
       #               (score_real[0] + score_real[1])

        # Print log data

        #text_is_predict_good = "Good"
        #if is_predict_good < config.model2_accuracy_of_a_good_prediction:
        #    text_is_predict_good = "NO Good"

        #if i < count:
        #    print("pred: " + str(score_pred) + "  real: " +
        #          str(score_real) + "\ndelta: " + str(score_delta) +
        #          " -> " + text_is_predict_good)

        # Save data
        #score_delta_array.append(score_delta)
        #if is_predict_good:
        #    good_pred_score_delta_array.append(score_delta)

        # Loop
        i += 1

    # Delta stats for ALL ( average, abs average, median )
    print("")
    delta_average = 0
    delta_abs_average = 0
    for val in score_delta_array:
        delta_average += val
        delta_abs_average += abs(val)
    delta_average /= len(score_delta_array)
    delta_abs_average /= len(score_delta_array)
    delta_median = statistics.median(score_delta_array)
    # Print
    print("ALL Average delta: " + str(delta_average))
    print("ALL Average ABS delta: " + str(delta_abs_average))
    print("ALL Median delta: " + str(delta_median))

    # Good and no good predictions
    print("")
    if len(good_pred_score_delta_array) == 0:
        print("Do not have a Good prediction (count == 0)")
    else:
        print("Good predictions precentage is : " +
              str(round(len(good_pred_score_delta_array) /
                        len(score_delta_array) * 100, 2)) +
              " %")

    # Delta stats for GOOD ( average, abs average, median )
    print("")
    delta_average = 0
    delta_abs_average = 0
    for val in good_pred_score_delta_array:
        delta_average += val
        delta_abs_average += abs(val)
    delta_average /= len(good_pred_score_delta_array)
    delta_abs_average /= len(good_pred_score_delta_array)
    delta_median = statistics.median(good_pred_score_delta_array)
    # Print
    print("GOOD Average delta: " + str(delta_average))
    print("GOOD Average ABS delta: " + str(delta_abs_average))
    print("GOOD Median delta: " + str(delta_median))


def evaluate_prediction_model3(tensor_predict, tensor_target, show_count):

    # Convert data type
    if type(tensor_predict) is torch.Tensor:
        tensor_predict = tensor_predict.tolist()
    if type(tensor_target) is torch.Tensor:
        tensor_target = tensor_target.tolist()

    # Calc delta
    delta_array = []
    good_predictions_count = 0
    i = 0
    for predict in tensor_predict:

        # Calc is a good prediction
        if tensor_target[i][0] > 0.5 and predict[0] > 0.5 or \
                tensor_target[i][0] < 0.5 and predict[0] < 0.5:
            good_predictions_count += 1

        # Calc delta
        delta_array.append(tensor_target[i][0] - predict[0])

        # Print
        if i < show_count:
            print(f"pred: {round(predict[0], 2)}  real: {round(tensor_target[i][0], 2)}")

        i += 1

    # Calc precdict_accuracy
    precdict_accuracy = round(good_predictions_count / len(tensor_predict) * 100, 2)

    # Calc delta accuracy
    delta_average = 0
    delta_abs_average = 0
    for val in delta_array:
        delta_average += val
        delta_abs_average += abs(val)
    delta_average = round(delta_average / len(delta_array), 2)
    delta_abs_average = round(delta_abs_average / len(delta_array), 2)
    delta_median = round(statistics.median(delta_array), 2)

    # Print
    print("")
    print(">>>> 5th gameset prediction <<<<")
    print("Good predictions: " + str(precdict_accuracy) + " %")
    print("Average delta: " + str(delta_average))
    print("Average ABS delta: " + str(delta_abs_average))
    print("Median delta: " + str(delta_median))