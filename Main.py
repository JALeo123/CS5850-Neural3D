import os
from os import path
import sys
import numpy as np
import random
import keras
from Data_Load_Process import load_data
from Data_Load_Process import rescale_mapping
from Data_Load_Process import generate_PDB
from Data_Load_Process import generate_log
from Neural_Models import new_Dense
from Evaluations import calc_metrics

def main():
    #Get Files
    args = sys.argv
    run_type = 0
    train_FISH = 1
    '''
    if(len(sys.argv) == 4):
        print("Running with Finetuning")
        data_path_train = "./Data/NeuralRun_Data/Matrix_Data/" + str(args[1])
        mapping_train = "./Data/NeuralRun_Data/Train_Structures/regular90/" + str(args[2])
        data_path_test = "./Data/NeuralRun_Data/Matrix_Data/" + str(args[3])
        test_file = str(args[3])
        run_type = 0
        if (path.exists(data_path_train) and path.exists(mapping_train) and path.exists(data_path_test)):
            print("Found All Input Files!")
        else:
            print("Error Finding Input Files")
            sys.exit()
    elif(len(sys.argv) == 2):
        print("Running without Finetuning")
        data_path_test = "./Data/NeuralRun_Data/Matrix_Data/" + str(args[1])
        test_file = str(args[1])
        run_type = 1
        if(path.exists(data_path_test)):
            print("Found All Input Files!")
        else:
            print("Error Finding Input Files")
            sys.exit()
    else:
        print("Wrong Parameters Set!")
        sys.exit()
    '''
    data_path_train = "./Data/NeuralRun_Data/Matrix_Data/regular90.txt"
    mapping_train = "./Data/NeuralRun_Data/Train_Structures/regular90/best_structure_regular90_IF.pdb"
    data_path_test = "./Data/NeuralRun_Data/Matrix_Data/regular70.txt"
    test_file = "regular70.txt"

    scale_factor = 100
    IF_alpha = 0.4
    epochs = 40
    batch_size = 20

    #Load All Data
    x_train, y_train, x_train_FISH, y_train_FISH, x_test, input_shape, scales_cal_values, matrix_table_test = load_data(data_path_train, data_path_test, mapping_train, scale_factor, IF_alpha, run_type)

    #Generate Models
    x_model = new_Dense(scale_factor+1, input_shape)
    y_model = new_Dense(scale_factor+1, input_shape)
    z_model = new_Dense(scale_factor+1, input_shape)

    #Train Models
    y_train_cat_x = keras.utils.to_categorical(y_train[0], scale_factor+1)
    y_train_cat_y = keras.utils.to_categorical(y_train[1], scale_factor+1)
    y_train_cat_z = keras.utils.to_categorical(y_train[2], scale_factor+1)
    y_train_cat_x_FISH = keras.utils.to_categorical(y_train_FISH[0], scale_factor+1)
    y_train_cat_y_FISH = keras.utils.to_categorical(y_train_FISH[1], scale_factor+1)
    y_train_cat_z_FISH = keras.utils.to_categorical(y_train_FISH[2], scale_factor+1)

    if(train_FISH == 1):
        x_model.fit(x_train_FISH, y_train_cat_x_FISH,
                      batch_size=5,
                      epochs=epochs,
                      verbose=1)
        y_model.fit(x_train_FISH, y_train_cat_y_FISH,
                      batch_size=5,
                      epochs=epochs,
                      verbose=1)
        z_model.fit(x_train_FISH, y_train_cat_z_FISH,
                      batch_size=5,
                      epochs=epochs,
                      verbose=1)

    if(run_type == 0):
        x_model.fit(x_train, y_train_cat_x,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)
        y_model.fit(x_train, y_train_cat_y,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)
        z_model.fit(x_train, y_train_cat_z,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

    #Predict Test Data
    val_x = x_model.predict_classes(x_test)
    val_y = y_model.predict_classes(x_test)
    val_z = z_model.predict_classes(x_test)

    #Unscale Data
    all_predictions = [val_x, val_y, val_z]
    all_predictions_scaled = rescale_mapping(all_predictions, scales_cal_values, matrix_table_test)

    #Generate PDB File
    output_file_message = "NEURAL 3D MODELING"
    generate_PDB(all_predictions_scaled, output_file_message)

    #Evaluations and Write to Log File
    metrics = calc_metrics(all_predictions_scaled, matrix_table_test)

    #WRITE EVALUATIONS
    generate_log(test_file, IF_alpha, scale_factor, metrics)

if __name__ == "__main__":
    main()