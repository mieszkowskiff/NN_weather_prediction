import numpy as np
import copy
import time

def mnist_normalize(X):
    return X / 255

def denormalize():
    
    return

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_matrix_derivative(x):
    temp = sigmoid_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def leaky_ReLU(x):
    return np.maximum(x, 0.5 * x)

def leaky_ReLU_derivative(x):
    def leaky_ReLU_derivative_lambda(x):
        if x <= 0:
            return 0.5
        else:
            return 1
    return np.array(list(map(leaky_ReLU_derivative_lambda, x))).reshape(-1, 1)

def leaky_ReLU_matrix_derivative(x):
    temp = leaky_ReLU_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - pow(tanh(x), 2)

def tanh_matrix_derivative(x):
    temp = tanh_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def softmax(x, T=1):
    return np.exp(x / T) / np.sum(np.exp(x / T), axis=0)

def softmax_matrix_derivative(x):
    temp = softmax(x)
    #print("temp: ", temp.shape)
    #print("temp: ", temp)
    m = -np.matmul(temp, temp.T)
    np.fill_diagonal(m, temp * (1 - temp))
    return m

def regression_data_normalization(x, l=-1, u=1, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x, axis=1, keepdims=True)
    if x_max is None:
        x_max = np.max(x, axis=1, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    return x * (u - l) + l, x_min, x_max

def regression_data_denormalization(x, x_min, x_max, l=-1, u=1):
    return (x - l) / (u - l) * (x_max - x_min) + x_min

def classification_data_normalization(x, mean=None, std=None):
    if mean is None:
        mean = np.array([np.mean(x, axis=1)]).T
    if std is None:
        std = np.array([np.std(x, axis=1)]).T
    return (x - mean) / std, mean, std

def classification_data_denormalization(x, mean, std):
    return x * std + mean

def one_hot_encoding(y):
    out = np.zeros((len(np.unique(y)), y.shape[0]))
    for i in range(y.shape[0]):
        out[int(y[i]), i] = 1
    return out

def one_hot_decoding(y):
    return np.argmax(y, axis=0)

def data_shuffle(x, y, classification=False):
    permute = np.random.permutation(x.shape[1])
    x = x[:,permute]
    y = y[:,permute]
    return x, y

class NeuralNetwork:
    def __init__(
            self,
            structure,
            biases_present=True,
            activation='sigmoid',
            last_layer_activation='softmax'):

        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.biases_present = biases_present

        self.weights = [np.array(np.random.uniform(-1, 1, (int(self.structure[i + 1]), int(self.structure[i]))), dtype = np.float32) for i in range(self.layers)]
        if self.biases_present:
            self.biases = [np.array(np.random.uniform(-1, 1, (int(self.structure[i + 1]), 1)), dtype = np.float32) for i in range(self.layers)]
        else:
            self.biases = [np.array(np.zeros((int(self.structure[i + 1]), 1)), dtype = np.float32) for i in range(self.layers)]

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

        elif activation == 'leaky_ReLU':
            self.activation = leaky_ReLU
            self.activation_derivative = leaky_ReLU_derivative

        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative

        if last_layer_activation == 'softmax':
            self.last_layer_activation = softmax
            self.last_layer_activation_derivative = softmax_matrix_derivative

        elif last_layer_activation == 'tanh':
            self.last_layer_activation = tanh
            self.last_layer_activation_derivative = tanh_matrix_derivative

        elif last_layer_activation == 'sigmoid':
            self.last_layer_activation = sigmoid
            self.last_layer_activation_derivative = sigmoid_matrix_derivative

        elif last_layer_activation == 'leaky_ReLU':
            self.last_layer_activation = leaky_ReLU
            self.last_layer_activation_derivative = leaky_ReLU_matrix_derivative

        self.neurons = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]
        #print(self.neurons)
        self.act_fun_neurons = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]
        self.act_fun_neurons.append(self.act_fun_neurons[-1])
        #print(self.act_fun_neurons)
        self.chain = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]

        self.weights_gradient = [np.zeros((int(self.structure[i + 1]), int(self.structure[i])), dtype = np.float32) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]

        self.BATCH_SIZE = 0
        self.LEARNING_RATE = 0
        self.NUMBER_OF_EPOCHS = 0

    def assure_input(self, input):
        assert type(input) is np.ndarray
        assert input.shape[0] == self.structure[0]

    def assure_output(self, output):
        assert type(output) is np.ndarray
        assert output.shape[0] == self.structure[-1]

    def __call__(self, input):
        data = copy.deepcopy(input)
        for i in range(self.layers - 1):
            data = self.activation(np.matmul(self.weights[i], data) + self.biases[i])
        return self.last_layer_activation(np.matmul(self.weights[self.layers - 1], data) + self.biases[self.layers - 1])

    def cost(self, input, output):
        return np.sum((self(input).reshape(-1) - output.reshape(-1)) ** 2)
    '''
    def shake(self, impact, ratio):
        for i in range(self.layers):
            self.weights
    ''' 

    def forward(self, input):
        self.neurons[0] = np.matmul(self.weights[0], input) + self.biases[0]
        self.act_fun_neurons[0] = self.activation(self.neurons[0])
        for i in range(1, self.layers):
            self.neurons[i] = np.matmul(self.weights[i], self.act_fun_neurons[i-1]) + self.biases[i]
            self.act_fun_neurons[i] = self.activation(self.neurons[i])
        self.act_fun_neurons[self.layers] = self.last_layer_activation(self.neurons[-1])
        return self.act_fun_neurons[self.layers]

    def calculate_chain(self, input, output):
        self.forward(input)
        # DONT DELETE OLD METHOD
        # DONT DELETE CODE BELOW
        # DONT DELETE
        '''
        start_time1 = time.time()
        
        self.chain[-1] = np.zeros((int(self.structure[-1]), int(self.BATCH_SIZE)), dtype = np.float32)
        self.chain[-1] = self.chain[-1].T

        for b in range(self.BATCH_SIZE):
            self.chain[-1][b] = np.matmul(
            self.last_layer_activation_derivative(self.neurons[-1].T[b].reshape(-1,1)),
            self.last_layer_activation(self.neurons[-1].T[b].reshape(-1,1)) - output.T[b].reshape(-1,1)
            ).reshape(-1)
            
        self.chain[-1] = self.chain[-1].T
        
        #print("old chain[-1] shape: ", self.chain[-1].shape)
        #print("old chain[-1]: ", self.chain[-1])
        time1 = time.time() - start_time1
        print("chain[-1] comput. time OLD METHOD:", time1)
        '''
        
#A[k] = self.last_layer_activation_derivative(self.neurons[-1].T[b].reshape(-1,1))   last x last
#B[k] = self.last_layer_activation(self.neurons[-1].T[b].reshape(-1,1)) - output.T[b].reshape(-1,1)).reshape(-1) last x 1

#self.chain[-1][k] = matmult(A[k], B[k])
        
        '''
        print("old: ", (self.last_layer_activation(self.neurons[-1].T[0].reshape(-1,1)) - output.T[0].reshape(-1,1)).shape)
        print("old 0: ", self.last_layer_activation(self.neurons[-1].T[0].reshape(-1,1)) - output.T[0].reshape(-1,1))
        print("old 1: ", self.last_layer_activation(self.neurons[-1].T[1].reshape(-1,1)) - output.T[1].reshape(-1,1))
        print("old 2: ", self.last_layer_activation(self.neurons[-1].T[2].reshape(-1,1)) - output.T[2].reshape(-1,1))

        print("B: ", (self.last_layer_activation(self.neurons[-1]) - output ).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1).shape )
        print("B: ", (self.last_layer_activation(self.neurons[-1]) - output ).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)  )
        
        print("old: ", self.last_layer_activation_derivative(self.neurons[-1].T[0].reshape(-1,1)).shape)
        print("old 0: ", self.last_layer_activation_derivative(self.neurons[-1].T[0].reshape(-1,1)))
        print("old 1: ", self.last_layer_activation_derivative(self.neurons[-1].T[1].reshape(-1,1)))
        print("old 2: ", self.last_layer_activation_derivative(self.neurons[-1].T[2].reshape(-1,1)))
        
        print("A: ", self.last_layer_activation_derivative(self.neurons[-1][:, 0].reshape(-1, 1)).shape)
        print("A: ", self.last_layer_activation_derivative(self.neurons[-1][:, 0].reshape(-1, 1)))
        print("A: ", self.last_layer_activation_derivative(self.neurons[-1][:, 1].reshape(-1, 1)))
        print("A: ", self.last_layer_activation_derivative(self.neurons[-1][:, 2].reshape(-1, 1)))
        '''
        
        #start_time2 = time.time()
        '''
        A = np.zeros((int(self.BATCH_SIZE), int(self.structure[-1]), int(self.structure[-1])))
        for i in range(self.BATCH_SIZE):
            A[i] = self.last_layer_activation_derivative(self.neurons[-1][:, i].reshape(-1, 1))

        #B = (self.last_layer_activation(self.neurons[-1]) - output).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)
        #self.last_layer_activation(self.neurons[-1]).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1) - output.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)

        self.chain[-1] = np.matmul(A, (self.last_layer_activation(self.neurons[-1]) - output).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)).T[0]
        
        #.reshape(int(self.structure[-1]), int(self.BATCH_SIZE))
        '''
        A = np.zeros((int(self.BATCH_SIZE), int(self.structure[-1]), int(self.structure[-1])))
        for i in range(self.BATCH_SIZE):
            A[i] = self.last_layer_activation_derivative(self.neurons[-1][:, i].reshape(-1, 1))

        #B = (self.last_layer_activation(self.neurons[-1]) - output).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)
        #self.last_layer_activation(self.neurons[-1]).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1) - output.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)

        self.chain[-1] = np.matmul(A, (self.act_fun_neurons[self.layers] - output).T.reshape(int(self.BATCH_SIZE), int(self.structure[-1]), 1)).T[0]
        #time2 = time.time() - start_time2
        #print("chain[-1] comput. time VECTOR PRODIGY:", time2)
        
        #print('SPEED UP: ', round(time1/time2 * 100, 2), '%')
        #print("")
        #print("new chain[-1] shape: ", self.chain[-1].reshape(int(self.structure[-1]), int(self.BATCH_SIZE)).shape)
        #print("new chain[-1]: ", self.chain[-1].reshape(int(self.structure[-1]), int(self.BATCH_SIZE)))

        #start_time_loop = time.time()
        for i in range(self.layers - 2, -1, -1):
            self.chain[i] = np.matmul(self.weights[i + 1].T, self.chain[i + 1]) * self.activation_derivative(self.neurons[i])
        #time_loop = time.time() - start_time_loop
        #print("loop time:", time_loop)

    def backward(self, input, output):
        
        self.calculate_chain(input, output)
        #start_time_back = time.time()
        self.weights_gradient[0] += np.matmul(self.chain[0], input.T/self.BATCH_SIZE*self.LEARNING_RATE)
        self.biases_gradient[0] += np.sum(self.chain[0], axis = 1).reshape(-1, 1)/self.BATCH_SIZE*self.LEARNING_RATE

        for i in range(1, self.layers):
            self.weights_gradient[i] += np.matmul(self.chain[i], self.act_fun_neurons[i-1].T/self.BATCH_SIZE*self.LEARNING_RATE)
            self.biases_gradient[i] += np.sum(self.chain[i], axis = 1).reshape(-1, 1)/self.BATCH_SIZE*self.LEARNING_RATE

        self.chain = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]
        #print("backward time: ", time.time() - start_time_back)

    def end_batch(self):
        #start_time_update = time.time()
        for i in range(self.layers):
            self.weights[i] -= self.weights_gradient[i]
            if self.biases_present:
                self.biases[i] -= self.biases_gradient[i]
        
        self.weights_gradient = [np.zeros((int(self.structure[i + 1]), int(self.structure[i])), dtype = np.float32) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((int(self.structure[i + 1]), 1), dtype = np.float32) for i in range(self.layers)]
        #print("update time: ", time.time() - start_time_update)

    def perform_training(self, X_train, Y_train, X_test, Y_test, batch_size=10, learning_rate=0.5, number_of_epochs=100, monitor_w = 2, monitor_b = 1):
        
        # training parameters
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.NUMBER_OF_EPOCHS = number_of_epochs

        num_of_batches = int(X_train.shape[1]/self.BATCH_SIZE)
        batches = [self.BATCH_SIZE*i for i in range(num_of_batches + 1)]
        
        num_of_breaks = 4
        break_len = int(num_of_batches/num_of_breaks)
        epoch_break_points = [i*break_len for i in range(num_of_breaks)]
        epoch_break_points.append(num_of_batches)

        costs = np.zeros((self.NUMBER_OF_EPOCHS*num_of_breaks))
        parameter_progress = np.zeros((monitor_w*self.layers, self.NUMBER_OF_EPOCHS*num_of_breaks))
        #parameter_gradient_progress = np.zeros((monitor_w*self.layers, self.NUMBER_OF_EPOCHS*num_of_breaks))

        monitor_w = int(monitor_w)
        weights_to_monitor = np.array(
            [ 
                [
                    i, np.random.randint(0, self.structure[i+1]), np.random.randint(0, self.structure[i])
                ] for i in range(self.layers) 
                for _ in range(monitor_w) 
            ]
        )

        for j in range(self.NUMBER_OF_EPOCHS):
            epoch_start_time = time.time()
            X_train, Y_train = data_shuffle(X_train, Y_train, True)
            print("Epoch #", j+1)
            
            # IN BATCH LOOP, CODE HAS TO BE MINIMIZED
            # CRUCIAL PART OF THE CODE FOR THE PERFORMANCE
            # DONT USE append(), DONT GET WEIGHTS VALUES EVERY BATCH, 
            # DONT EVALUATE COST EVERY BATCH
            for b in range(len(epoch_break_points) - 1):
                print("Batch #", epoch_break_points[b], "/", num_of_batches)
                for i in range(epoch_break_points[b], epoch_break_points[b+1]):
                    #batch_start_time = time.time()
                    self.backward(X_train[:,batches[i]:batches[i+1]], Y_train[:,batches[i]:batches[i+1]])
                    self.end_batch()
                    #time_batch = time.time() - batch_start_time
                    #print("batch time: ", time_batch)
                
                # NN learning monitor
                costs[j*num_of_breaks + b] = self.cost(X_test, Y_test)

                for lay in range(self.layers):
                    for m in range(monitor_w): 
                        watch_weight = weights_to_monitor[lay*monitor_w+m]
                        parameter_progress[lay*monitor_w + m][j*num_of_breaks + b] = self.weights[watch_weight[0]][watch_weight[1]][watch_weight[2]]
                        #parameter_gradient_progress[lay+m][j*num_of_breaks + b] = self.weights_gradient[watch_weight[0]][watch_weight[1]][watch_weight[2]]

            print("loss fun. on test: ", costs[(j + 1)*num_of_breaks - 1])
            print("epoch time: ", time.time() - epoch_start_time)

        return costs, parameter_progress