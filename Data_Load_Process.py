import math
import sys
import numpy as np

# Load Train and Test Data
def load_data(contact_matrix_path_train, contact_matrix_path_test, mapping_train_path, scale_factor, IF_alpha, run_type):
    x_train = []
    y_train = []
    x_test = []
    x_train_FISH = []
    y_train_FISH = []

    #Process Train Data
    if(run_type == 0):
        file_train = open(contact_matrix_path_train, "r")
        file_train_mapping = open(mapping_train_path, "r")
        matrix_table_train = matrix_to_table(file_train, "Train")
        matrix_table_train = IF_to_distance(matrix_table_train, IF_alpha)
        train_labels, scales_cal_values = scale_mapping(file_train_mapping, scale_factor)
        x_train = np.array(matrix_table_train)
        y_train_x = []; y_train_y = []; y_train_z = []
        for i in range(len(x_train)):
            y_train_x.append(train_labels[0][matrix_table_train[i][0] - 1])
            y_train_y.append(train_labels[1][matrix_table_train[i][0] - 1])
            y_train_z.append(train_labels[2][matrix_table_train[i][0] - 1])
        y_train = [y_train_x, y_train_y, y_train_z]
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    else:
        scales_cal_values = ""

    x_train_FISH = matrix_to_table_FISH()
    x_train_FISH = np.array(IF_to_distance(x_train_FISH, IF_alpha))
    x_train_FISH = x_train_FISH.reshape(x_train_FISH.shape[0], x_train_FISH.shape[1], 1)
    y_train_FISH = np.array(scale_mapping_FISH(scale_factor))

    file_test = open(contact_matrix_path_test, "r")
    matrix_table_test = matrix_to_table(file_test, "Test")
    matrix_table_test = IF_to_distance(matrix_table_test, IF_alpha)
    x_test = np.array(matrix_table_test)

    #Reshape Train, Test Data
    return_shape = (x_test.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    if (run_type == 0):
        file_train.close()
        file_train_mapping.close()

    return x_train, y_train, x_train_FISH, y_train_FISH, x_test, return_shape, scales_cal_values, matrix_table_test

# Convert from NxN matrix to matrix table
def matrix_to_table(file, type):
    lines = file.readlines()
    num_lines = len(lines)
    write_to_file = True

    matrix_lists = []
    for line in lines:
        line = line.strip('\n')
        l_new = line.split('\t')

        for i in range(len(l_new)):
            l_new[i] = int(l_new[i])
        matrix_lists.append(l_new.copy())

    matrix_table = []
    for i in range(0, num_lines):
        for j in range(0, num_lines):
            row = []
            row.append(i + 1); row.append(j + 1); row.append(matrix_lists[i][j])
            if (i != j):
                matrix_table.append(row.copy())

    # Raw Matrix Table to File
    if (write_to_file == True):
        out_file = open("./Generated_Outputs/Contact_Matrix_Table_" + type + ".txt", "w")
        for row in matrix_table:
            out_file.write(str(row[0]) + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n')
        out_file.close()

    return matrix_table

# Convert from NxN matrix to matrix table for FISH data
def matrix_to_table_FISH():
    file = open("./Data/NeuralRun_Data/FISH/GM06990/covn200kchr22.txt", "r")
    lines = file.readlines()
    num_lines = len(lines)

    matrix_lists = []
    for line in lines:
        line = line.strip('\n')
        l_new = line.split('\t')
        l_new = l_new[1:4]

        for i in range(len(l_new)):
            l_new[i] = int(l_new[i])
        matrix_lists.append(l_new.copy())

    for i in range(len(matrix_lists)):
        matrix_lists[i][0] = i+1
        matrix_lists[i][1] = i+2

    matrix_table = matrix_lists[:-1]
    file.close()

    return matrix_table

#Perform IF to distance conversion based on alpha value
def IF_to_distance(matrix_table, alpha):
    #temp = 0
    for i in range(len(matrix_table)):
        if(matrix_table[i][2] != 0):
     #       if((1 / math.pow(matrix_table[i][2], alpha)) > 1 or (1 / math.pow(matrix_table[i][2], alpha)) < 0.1):
     #           print((1 / math.pow(matrix_table[i][2], alpha)))
     #           print(matrix_table[i])
     #           temp += 1
            matrix_table[i][2] = (1 / math.pow(matrix_table[i][2], alpha))
        else:
            matrix_table[i][2] = 1
    #print(temp)

    return matrix_table

#Extract 3D mapping for Training and Validation, Scale factor applied
def scale_mapping(file_train_mapping, scale_factor):
    lines = file_train_mapping.readlines()
    select_lines = []
    for i in range(len(lines)):
        if(lines[i].find("ATOM") != -1): #Specific for PDB file
            read_line = lines[i]
            read_line = read_line.strip('\n')
            read_line = read_line.split(' ')
            l_new = []
            for j in range(len(read_line)):
                if(read_line[j] != ''):
                    l_new.append(read_line[j])
            select_lines.append(l_new[5:8])

    #Apply scaling
    x_labels = []
    y_labels = []
    z_labels = []
    for i in range(len(select_lines)):
        x_labels.append(float(select_lines[i][0]))
        y_labels.append(float(select_lines[i][1]))
        z_labels.append(float(select_lines[i][2]))

    scale_calc_x = scale_factor / max(x_labels)
    scale_calc_y = scale_factor / max(y_labels)
    scale_calc_z = scale_factor / max(z_labels)
    for i in range(len(x_labels)):
        x_labels[i] = round(x_labels[i] * scale_calc_x)
        y_labels[i] = round(y_labels[i] * scale_calc_y)
        z_labels[i] = round(z_labels[i] * scale_calc_z)

    data_labels = [x_labels, y_labels, z_labels]
    scales_cal_values = [scale_calc_x, scale_calc_y, scale_calc_z]

    return data_labels, scales_cal_values

#Extract 3D mapping for Training and Validation, Scale factor applied for FISH
def scale_mapping_FISH(scale_factor):
    file = open("./Data/NeuralRun_Data/FISH/GM06990/gmnn200kchr22.xyz", "r")
    lines = file.readlines()
    select_lines = []
    for i in range(len(lines)):
        read_line = lines[i]
        read_line = read_line.strip('\n')
        read_line = read_line.split(' ')
        l_new = []
        for i in read_line:
            l_new.append(float(i))
        select_lines.append(l_new.copy())

    #Apply scaling
    x_labels = []
    y_labels = []
    z_labels = []
    for i in range(len(select_lines)):
        x_labels.append(float(select_lines[i][0]))
        y_labels.append(float(select_lines[i][1]))
        z_labels.append(float(select_lines[i][2]))

    scale_calc_x = scale_factor / max(x_labels)
    scale_calc_y = scale_factor / max(y_labels)
    scale_calc_z = scale_factor / max(z_labels)
    for i in range(len(x_labels)):
        x_labels[i] = round(x_labels[i] * scale_calc_x)
        y_labels[i] = round(y_labels[i] * scale_calc_y)
        z_labels[i] = round(z_labels[i] * scale_calc_z)

    data_labels = [x_labels, y_labels, z_labels]
    file.close()

    return data_labels

#Rescale Predicted Data
def rescale_mapping(all_predictions, scales_cal_values, matrix_table_test):
    x_predictions = list(all_predictions[0])
    y_predictions = list(all_predictions[1])
    z_predictions = list(all_predictions[2])

    count = 0
    for i in range(len(matrix_table_test)):
        if(matrix_table_test[i][0] == 1):
            count += 1

    for i in range(len(x_predictions)):
        x_predictions[i] = x_predictions[i] / scales_cal_values[0]
        y_predictions[i] = y_predictions[i] / scales_cal_values[1]
        z_predictions[i] = z_predictions[i] / scales_cal_values[2]

    x_pred_final = []; y_pred_final = []; z_pred_final = []
    count2 = 0
    sum_x = 0; sum_y = 0; sum_z = 0
    for i in range(len(x_predictions)):
        if(count2 < count-1):
            sum_x += x_predictions[i]
            sum_y += y_predictions[i]
            sum_z += z_predictions[i]
            count2 += 1
        else:
            x_pred_final.append(sum_x / count)
            y_pred_final.append(sum_y / count)
            z_pred_final.append(sum_z / count)
            count2 = 0
            sum_x = 0; sum_y = 0; sum_z = 0

    rescaled_predictions = [x_pred_final, y_pred_final, z_pred_final]

    return rescaled_predictions

#Generate PDB file
def generate_PDB(all_predictions, output_file_message):
    out_pdb = open("./Generated_Outputs/OutputPDB.pbd", "w")

    out_pdb.write(output_file_message + "\n")
    for i in range(1, len(all_predictions[0])):
        count_str = ""
        count_str2 = ""
        if(i <= 9):
            count_str = "      " + str(i)
            count_str2 = "          " + str(i)
        elif(i <= 99 and i > 9):
            count_str = "     " + str(i)
            count_str2 = "         " + str(i)
        else:
            count_str = "    " + str(i)
            count_str2 = "        " + str(i)
        out_pdb.write("ATOM" + count_str + " " + "CA" + " " + "MET" + " " + "A" + count_str2 + " "
                      + str(round(all_predictions[0][i-1],3)) + "  " + str(round(all_predictions[1][i-1],3)) + "  " + str(round(all_predictions[2][i-1],3)) + "  \n")

    for i in range(1, len(all_predictions[0])):
        count_str = ""
        count_str2 = ""
        if(i <= 9):
            count_str = "    " + str(i)
            if(i == 9):
                count_str2 = "   " + str(i+1)
            else:
                count_str2 = "    " + str(i + 1)
        elif(i <= 99 and i > 9):
            count_str = "   " + str(i)
            if(i == 99):
                count_str2 = "  " + str(i+1)
            else:
                count_str2 = "   " + str(i + 1)
        else:
            count_str = "  " + str(i)
            count_str2 = "  " + str(i+1)
        out_pdb.write("CONNECT" + count_str + count_str2 + "\n")

    out_pdb.write("END")

    out_pdb.close()

#Generate Log File
def generate_log(test_file, IF_alpha, scale_factor, metrics):
    out_log = open("./Generated_Outputs/Output.log", "w")

    out_log.write("Input File: " + test_file + "\n")
    out_log.write("IF Alpha Value: " + str(IF_alpha) + "\n")
    out_log.write("Scale Factor Value: " + str(scale_factor) + "\n")
    out_log.write("AVG RMSE: " + str(metrics[0]) + "\n")
    out_log.write("AVG Spearman Correlation Dist vs. Reconstructed Dist: " + str(metrics[1]) + "\n")
    out_log.write("AVG Pearson Correlation Dist vs. Reconstructed Dist: " + str(metrics[2]) + "\n")

    print("Input File: " + test_file)
    print("IF Alpha Value: " + str(IF_alpha))
    print("Scale Factor Value: " + str(scale_factor))
    print("AVG RMSE: " + str(metrics[0]))
    print("AVG Spearman Correlation Dist vs. Reconstructed Dist: " + str(metrics[1]))
    print("AVG Pearson Correlation Dist vs. Reconstructed Dist: " + str(metrics[2]))

    out_log.close()
