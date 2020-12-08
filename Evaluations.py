import numpy as np
import math
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import sys

def calc_metrics(all_predictions_scaled, matrix_table_test, method):
    dist_actual = []
    dist_actual_actual = []
    x = []; y = []; z = []
    if(method != "Neural_3D_Modeling"):
        file = open("./Data/NeuralRun_Data/Matrix_Data/regularstructre.txt")
        lines = file.readlines()
        for line in lines:
            line.strip('\n')
            l_new = line.split(',')
            x.append(float(l_new[0]))
            y.append(float(l_new[1]))
            z.append(float(l_new[2]))
        dist_actual = [x, y, z]
        for i in range(len(dist_actual[0])):
            for j in range(i + 1, len(dist_actual[0])):
                sub_x = math.pow((dist_actual[0][j] - dist_actual[0][i]), 2)
                sub_y = math.pow((dist_actual[1][j] - dist_actual[1][i]), 2)
                sub_z = math.pow((dist_actual[2][j] - dist_actual[2][i]), 2)
                distance = math.sqrt(sub_x + sub_y + sub_z)
                dist_actual_actual.append(distance)
        dist_actual = dist_actual_actual
    else:
        for i in range(len(matrix_table_test)):
            if(matrix_table_test[i][1] > matrix_table_test[i][0]):
                dist_actual.append(matrix_table_test[i][2])

    dist_pred = []
    for i in range(len(all_predictions_scaled[0])):
        for j in range(i+1, len(all_predictions_scaled[0])):
            sub_x = math.pow((all_predictions_scaled[0][j] - all_predictions_scaled[0][i]), 2)
            sub_y = math.pow((all_predictions_scaled[1][j] - all_predictions_scaled[1][i]), 2)
            sub_z = math.pow((all_predictions_scaled[2][j] - all_predictions_scaled[2][i]), 2)
            distance = math.sqrt(sub_x + sub_y + sub_z)
            dist_pred.append(distance)

    Pearson_val = pearsonr(dist_actual, dist_pred)[0]
    Spearman_val = spearmanr(dist_actual, dist_pred)[0]
    RMSE_val = math.sqrt(mean_squared_error(dist_actual, dist_pred))
    metrics = [RMSE_val, Pearson_val, Spearman_val]

    return metrics
