#importing the nessecary libraries for our Nueral Network Model
import numpy as np
import csv #import csv to read and write on csv-files
from scipy.special import logsumexp
class MLP():
    """A Multilayer Perceptron class.
    """  
    def __init__(self,x):
        """Constructor for the MLP.Takes a variable number of hidden layers and desired error """
       # print('you made a class')
        self.num_inputs = num_inputs=35
        self.hidden_layers=x
        self.num_outputs = num_outputs=10          
        #Creating a generic representation of the layers
        layers = [num_inputs] + x + [num_outputs]
        # Creating random connection weights for the layers        
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.uniform(-2.4/35,2.4/35,(layers[i], layers[i + 1]))
            weights.append(w)
        self.weights = weights
        
        bias = []
        de_bias=[]
        threshold=self.hidden_layers+[num_outputs]
        
        for i in range(len(threshold)):
          
           h=np.empty(threshold[i])
           de_threshold=np.empty(threshold[i])
           h.fill(np.random.uniform(-2.4/35,2.4/35))
           de_threshold.fill(0)
           bias.append(h)
           de_bias.append(de_threshold)
        self.bias = bias
        self.de_bias=de_bias
        
        # save derivatives per layer
        derivatives = []
        
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
    def __del__(self):    
          print('Destructor called, MLP destroyed')
        #return self.hidden_layers
    def forward_propagate_relu(self, inputs):
        """Computes forward propagation of the network based on input signals."""
        # the input layer activation is just the input itself
        activations = inputs        
        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)
            if i==(len(self.weights)-1):                         
              activations=self.linear(net_inputs+self.bias[i])      
              activations=self.soft_max(activations)
              # save the activations for backpropogation
              self.activations[i+1] = activations               
             # save the activations for backpropogation            
            else:
                # apply Relu activation function
                activations = self.Relu(net_inputs+self.bias[i])
                # save the activations for backpropogation
                self.activations[i+1] = activations                         
        # return output layer activation
        return activations

    def back_propagate_relu(self, error,target,output):
        """Backpropogates an error signal."""
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
             if i==len(self.derivatives)-1:
                activations = self.activations[i+1]
                delta =np.dot( error , self.softmax_dash(activations))                              
                delta_re=np.array(delta)   
                #reshape delta as to have it as a 2d array
                delta_re = delta_re.reshape(delta_re.shape[0],-1).T
               # get activations for current layer
                current_activations = self.activations[i]
                current_activations=np.array(current_activations)
                current_activations=current_activations.reshape(current_activations.shape[0],-1)
               # save derivative after applying matrix multiplication
                self.de_bias[i]=delta
                self.derivatives[i] = np.dot(current_activations, delta_re)
               # backpropogate the next error
                error = np.dot(delta, self.weights[i].T) 
             else:  
             # get activation for previous layer
              activations = self.activations[i+1]
            # apply sigmoid derivative function
              delta = error * self._Relu_derivative((activations))
              delta_re=np.array(delta)          
            # reshape delta as to have it as a 2d array
              delta_re = delta_re.reshape(delta_re.shape[0],-1).T           
            # get activations for current layer
              current_activations = self.activations[i]            
              current_activations=np.array(current_activations)           
            # reshape activations as to have them as a 2d column matrix
              current_activations = current_activations.reshape(current_activations.shape[0],-1)          
            # save derivative after applying matrix multiplication
              self.de_bias[i]=delta
              self.derivatives[i] = np.dot(current_activations, delta_re)
            # backpropogate the next error
              error = np.dot(delta, self.weights[i].T)
            
    def data_training_relu(self,epochs, learning_rate,desired_error=0.05):
        self.error=desired_error
        with open('output_data2.csv', newline = '') as file:
            reader_output = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)       
            Training_data_output = []
            for  row in reader_output:
                     Training_data_output.append(row[:]) 
        with open('input_data2.csv', newline = '') as file:
                reader_data = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
                Training_data_csv=[]
                for  rowss in reader_data:
                        Training_data_csv.append(rowss[:]) 
        plotting_array=[]                         
        for i in range(epochs):
            total_cross_entrpy=0 
            sum_errors = 0
            # iterate through all the training data
            for j, input in enumerate(Training_data_csv):
                target = Training_data_output[j]
                # activate the network!
                output = self.forward_propagate_relu(input)
                error = target - output                
                self.back_propagate_relu(error,target,output)
                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)
                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)
                total_cross_entrpy+=self.cross_entropy(target,output)
              # Epoch complete, report the training error
              
            if (sum_errors / len(Training_data_csv))< self.error:
                break
        
            # Epoch complete, report the training error
            print("Cross entropy: {} at epoch {}".format(total_cross_entrpy / len(Training_data_csv),i+1))
            print("Mean Squared Error: {} at epoch {}".format(sum_errors / len(Training_data_csv),i+1))
            plotting_array.append(sum_errors / len(Training_data_csv))
        print("Cross entropy: {} at epoch {}".format(total_cross_entrpy / len(Training_data_csv),i))
        print("Mean Squared Error: {} at epoch {}".format(sum_errors / len(Training_data_csv),i+1))
        print("Training complete!")
        print("=====")
        self.plot_array=plotting_array
        self.final_mse= sum_errors / len(Training_data_csv) 
        self.num_of_epechs=i+1
        self.final_cross=total_cross_entrpy / len(Training_data_csv)
    def forward_propagate_tanh(self, inputs):
         """Computes forward propagation of the network based on input signals."""
         # the input layer activation is just the input itself
         activations = inputs
         # save the activations for backpropogation
         self.activations[0] = activations
         # iterate through the network layers
         for i, w in enumerate(self.weights):
             # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)  
             # apply activation function            
            if i==(len(self.weights)-1):                          
              activations=self.linear(net_inputs+self.bias[i])              
              activations=self.soft_max(activations)
              # save the activations for backpropogation 
              self.activations[i+1] = activations                                       
            else:
                 activations = self.Tanh(net_inputs+self.bias[i])
                 # save the activations for backpropogation 
                 self.activations[i+1] = activations
         # return output layer activation
         return activations    
    def back_propagate_tanh(self, error,target,output):
        """Backpropogates an error signal"""
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
             if i==len(self.derivatives)-1:
                activations = self.activations[i+1]
                delta =np.dot( error , self.softmax_dash(activations))
                delta_re=np.array(delta)   
                #reshape delta as to have it as a 2d array
                delta_re = delta_re.reshape(delta_re.shape[0],-1).T
               # get activations for current layer
                current_activations = self.activations[i]
                current_activations=np.array(current_activations)
                current_activations=current_activations.reshape(current_activations.shape[0],-1)
               # save derivative after applying matrix multiplication
                self.de_bias[i]=delta
                self.derivatives[i] = np.dot(current_activations, delta_re)
               # backpropogate the next error
                error = np.dot(delta, self.weights[i].T) 
             else:  
             # get activation for previous layer
              activations = self.activations[i+1]
            # apply sigmoid derivative function
              delta = error * self._Tanh_derivative(activations)
              delta_re=np.array(delta)          
            # reshape delta as to have it as a 2d array
              delta_re = delta_re.reshape(delta_re.shape[0],-1).T           
            # get activations for current layer
              current_activations = self.activations[i]            
           # print('this currnet activ',current_activations)
              current_activations=np.array(current_activations)           
            # reshape activations as to have them as a 2d column matrix
              current_activations = current_activations.reshape(current_activations.shape[0],-1)          
            # save derivative after applying matrix multiplication
              self.de_bias[i]=delta
              self.derivatives[i] = np.dot(current_activations, delta_re)
            # backpropogate the next error
              error = np.dot(delta, self.weights[i].T)
    def data_training_tanh(self,epochs, learning_rate,desired_error=0.05):
            self.error=desired_error
            with open('output_data2.csv', newline = '') as file:
                reader_output = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)       
                Training_data_output = []               
                for  row in reader_output:
                         Training_data_output.append(row[:]) 
            with open('input_data2.csv', newline = '') as file:                  
                    reader_data = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)                  
                    Training_data_csv=[]
                    for  rowss in reader_data:
                            Training_data_csv.append(rowss[:]) 
            plotting_array=[]             
            for i in range(epochs):
                total_cross_entrpy=0 
                sum_errors = 0
                
                # iterate through all the training data
                for j, input in enumerate(Training_data_csv):
                    target = Training_data_output[j]
                    # activate the network!
                    output = self.forward_propagate_tanh(input)                   
                    error = target - output
                    self.back_propagate_tanh(error,target,output)
                    # now perform gradient descent on the derivatives
                    # (this will update the weights
                    self.gradient_descent(learning_rate)
                    # keep track of the MSE for reporting later
                    sum_errors += self._mse(target, output)
                    total_cross_entrpy+=self.cross_entropy(target,output)
                  # Epoch complete, report the training error
                if (sum_errors / len(Training_data_csv))< self.error:
                    break
               
                print("Cross entropy: {} at epoch {}".format(total_cross_entrpy / len(Training_data_csv),i+1))
                print("Mean Squared Error: {} at epoch {}".format(sum_errors / len(Training_data_csv),i+1))
                plotting_array.append(sum_errors / len(Training_data_csv))
            print("Cross entropy: {} at epoch {}".format(total_cross_entrpy / len(Training_data_csv),i))
            print("Mean Squared Error: {} at epoch {}".format(sum_errors / len(Training_data_csv),i+1))
            print("Training complete!")
            print("=====")
            self.plot_array=plotting_array
            self.final_mse= sum_errors / len(Training_data_csv) 
            self.num_of_epechs=i+1
            self.final_cross=total_cross_entrpy / len(Training_data_csv)
    def gradient_descent(self, learningRate):
        """Learns by descending the gradient"""
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            bias=self.bias[i]
            de_bias=self.de_bias[i]
            weights += derivatives * learningRate
            bias+=  de_bias* learningRate
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        y = 1.0 / (1 + np.exp(-x))
        return y
    
    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function"""
        return x * (1.0 - x)
  
    def Tanh(self, x):
        """Tanh activation function"""
        y=(2.0/(1+np.exp(-2*x)))-1.0
        return y
    
    def _Tanh_derivative(self, x):
        """Tanh derivative function"""       
        return (1.0-x**2)
    
    def Relu(self, x):
        """Relu activation function"""
        return np.maximum(0.0,x)
        
    def _Relu_derivative(self, x):
        """Relu derivative function"""
        x[x<=0] = 0
        x[x>0] = 1
        return x        

    def _mse(self, target, output):
        """Mean Squared Error loss function"""
        return np.average((target - output) ** 2)
    
    def  cross_entropy(self,y_true, y_pred):
        CT=list(zip(y_pred, y_true))   
        loss=0
        for entry in CT:
            predict=entry[0]
            actual=entry[1]
            loss+=-(actual*np.log(logsumexp(predict)))
                    #+(1-actual*math.log10(1-predict)))
        return loss
    def cross_ED(self,y_true, y_pred):
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
        return (-(y_true/y_pred)+((1-y_true)/1-y_pred))
    def soft_max(self,x):
        """ applies softmax to an input x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def softmax_dash(self,x):  
        # Softmax derivative
       I = np.eye(x.shape[0])       
       return self.soft_max(x) * (I - self.soft_max(x).T)
   
    def linear(self,x):        
       return x 
   
    def derev_linear(self,x):
       return 1