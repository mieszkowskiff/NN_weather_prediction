import read_data
import ai_np
import numpy as np
import itertools
from pandas import read_csv
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors



if __name__ == "__main__":
    print("Loading data...")
    
    path = "./clean_norm_data/new_feats/"

    X_train = read_csv(path + 'X_train.csv', sep=",", header=None, dtype=np.float32)
    Y_train = read_csv(path + 'Y_train.csv', sep=",", header=None, dtype=np.float32)
    
    X_test = read_csv(path + 'X_test.csv', sep=",", header=None, dtype=np.float32)
    Y_test = read_csv(path + 'Y_test.csv', sep=",", header=None, dtype=np.float32)
    # read params to denormalize
    mean = read_csv(path + 'mean.csv', header=None, dtype=np.float32)
    min = read_csv(path + 'min.csv', header=None, dtype=np.float32)
    max = read_csv(path + 'max.csv', header=None, dtype=np.float32)

    X_train = np.array(X_train.apply(pd.to_numeric, errors='coerce'))
    Y_train = np.array(Y_train.apply(pd.to_numeric, errors='coerce'))
    
    X_test = np.array(X_test.apply(pd.to_numeric, errors='coerce'))
    Y_test = np.array(Y_test.apply(pd.to_numeric, errors='coerce'))

    mean = np.array(mean.apply(pd.to_numeric, errors='coerce'))
    min = np.array(min.apply(pd.to_numeric, errors='coerce'))
    max = np.array(max.apply(pd.to_numeric, errors='coerce'))

    #print(min)
    
    if False:
        # for tests, reduce the dataset to 'end' instances
        # test - true, proper run - false
        end = 104
        X_train = X_train[0:end]
        Y_train = Y_train[0:end]
        X_test = X_test[0:end]
        Y_test = Y_test[0:end]
    

    print("Data loaded.")
    print("Processing data...")

    num_train_images = X_train.shape[0]
    num_test_images = X_test.shape[0]

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    # choose, prediction for the 2nd day
    # or for the 1st day
    if False:
        Y_train = Y_train[2:-1]
        Y_test = Y_test[2:-1]
    else:
        Y_train = Y_train[0:2]
        Y_test = Y_test[0:2]

    print("Data processed.")

    print("         feat. num.")
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  ' + str(X_test.shape))
    print('Y_test:  ' + str(Y_test.shape))
    print("Shapes should always be 2D matrices. Even if it means that 1st dimension is of size 1.")

    print("Creating neural network...")

    input = X_train.shape[0]
    output = Y_train.shape[0]

    if False:
        nn = ai_np.NeuralNetwork(
            structure= [input, 4096, 2048, 2048, 1024, 64, output], 
            activation='sigmoid', 
            last_layer_activation='tanh',
            biases_present = True
        )
        print("Neural network created.")
    else:
        path_model = "./models/2nd_iter/"
        path_weights = path_model + 'weights-new.npz'
        path_biases = path_model + 'biases-new.npz'
        weights = np.load(path_weights)
        biases = np.load(path_biases)

        act = 'sigmoid'
        last_layer_act = 'tanh'
        num_hidden_layers = len(weights) - 1
        structure = []
        
        structure.append(X_test.shape[0])
        for array_name in weights.files:
            structure.append(len(weights[array_name]))  
        
        nn = ai_np.NeuralNetwork(
            structure=structure, 
            biases_present = True,
            activation = act, 
            last_layer_activation = last_layer_act
        )
        nn.weights = [weights[arr_name] for arr_name in weights.files]
        nn.biases = [biases[arr_name] for arr_name in biases.files]

        print("Neural network model successfully recreated.")

    print("Beginning NN training.")

    # number of randomly selected weights from each layer user wishes to monitor
    monitor_w = 1

    # planning the trainning for the NN
    #   each row represents one stage of the training
    #       numbers in each row correspond to: batch size, learning rate 
    #       and number of epochs for the particular stage 
    training_plan = [
        [32, 0.1, 3]
    ]

    '''
    [10, 0.1, 5],
        [8, 0.05, 8],
        [4, 0.05, 5],
        [2, 0.01, 5]
    ,
        [8, 0.05, 3],
        [4, 0.005, 5]
    
    ,
        [16, 0.05, 0],
        [8, 0.005, 0]
    '''

    costs = []
    parameter_progress = []
    for stage in range(len(training_plan)):
        print("Stage #", stage + 1, '/', len(training_plan))
        c, p = nn.perform_training(
            X_train, 
            Y_train, 
            X_test, 
            Y_test,
            batch_size=training_plan[stage][0], 
            learning_rate=training_plan[stage][1], 
            number_of_epochs=training_plan[stage][2],
            monitor_w=monitor_w
        )
        costs.append(c)
        parameter_progress.append(p)
        #print(np.array(c).shape)
        #print(np.array(p).shape)

    costs = np.array(list(itertools.chain(*costs)))
    parameter_progress = np.array(list(itertools.chain(*parameter_progress)))
    #print(costs.shape)
    #print(parameter_progress.shape)


    print("NN training finished.")
    print("Validation...")

    Y_pred = nn.forward(X_test)
   
    # normalization params, normalizing to [-1, 1]
    lower = -1
    upper = 1

    print("Y_test shape: ", Y_test.shape)
    temp_test = Y_test[0]
    wind_test = Y_test[1]

    temp_test = (temp_test - lower)*(max[0] - min[0])/(upper - lower) + min[0]
    wind_test = (wind_test - lower)*(max[3] - min[3])/(upper - lower) + min[3]

    print("Temp. test:")
    print(temp_test[0:10])
    print(temp_test[10:20])
    print("Wind test:")
    print(wind_test[0:10])
    print(wind_test[10:20])
    
    print("Y_pred shape: ", Y_pred.shape)
    temp_pred = Y_pred[0]
    wind_pred = Y_pred[1]

    temp_pred = (temp_pred - lower)*(max[0] - min[0])/(upper - lower) + min[0]
    wind_pred = (wind_pred - lower)*(max[3] - min[3])/(upper - lower) + min[3]
    
    print("Temp. prediction:")
    print(temp_pred[0:10])
    print(temp_pred[10:20])
    print("Wind prediction:")
    print(wind_pred[0:10])
    print(wind_pred[10:20])

    #acc=sum(Y_test==Y_pred)
    #print('Accuracy: ', acc/len(Y_test) * 100, '%')

    plot_path = './plots/'
    if True:
        fig, ax = plt.subplots()
        ax.scatter(range(len(costs)), [cost for cost in costs], c='b', s=10, label='Cost')
        
        plt.title("Cost over epochs")
        plt.legend()
        ax.grid(True)
        
        plt.savefig(plot_path + 'cost.png')
        #plt.show()
        
        for p in range(nn.layers*monitor_w):
            fig, ax = plt.subplots()
            ax.scatter(range(len(parameter_progress[p])), np.array(parameter_progress[p]), c='b', s=10, label='parameter')
            
            plt.title("Chosen parameters over epochs")
            plt.legend()
            ax.grid(True)
            
            plt.savefig(plot_path + 'w' + str(p) + '.png')
            #plt.show()

    np.savez(f'./models/3rd_iter/weights-new.npz', *nn.weights)
    np.savez(f'./models/3rd_iter/biases-new.npz', *nn.biases)
