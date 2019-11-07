import numpy as np
from Activation import *
class Layer(object):
    def __init__(self,id,input_dim,dim,init_type="uniform",type="Input",activation="None"):
        #setting layer infomation in the network
        self.init_type = init_type
        self.dim = dim
        self.input_dim = input_dim
        self.id = id
        self.type = type
        self.activation = activation
        self.meta_info = {"id":self.id,"Name":self.type,"Inputdim":input_dim,"Outputdim":dim,"Number_of_Parameters":0,"Activation":self.activation}
        
    def initialize(self):
        #Derived Classes implement this method
        pass  

    def set_weights(self,weights,bias):
        self.weights = weights
        self.bias = bias

    def compute(self,prev_input):
        #Derived Classes implement this method
        return prev_input
    def compute_gradient(self,next_gradient):
        #Derived Classes implement this method
        return None,None
    def update_weights(self,learning_rate):
        #Derived Classes implement this method
        return       

class Dense(Layer):
    def __init__(self, id,input_dim,dim,init_type='standard', type='Dense',activation="relu"):
        #initialize dense layer
        super().__init__(id,input_dim,dim,init_type=init_type, type=type,activation=activation)
        self.compute_layer_info()    

    def compute_layer_info(self):
        #generate Dense layer information and store it
        self.meta_info["Outputdim"] = self.dim
        self.meta_info["Activation"] = self.activation
        self.weights_dim = (self.input_dim[1] , self.dim[1])   
        self.bias_dim = self.dim
        self.meta_info["Number_of_Parameters"] = self.weights_dim[0] * self.weights_dim[1] + self.bias_dim[1]

    def initialize(self):
        #initialize the matrix
        if self.init_type == "uniform" : 
            self.weights = np.random.uniform(size=self.weights_dim)
            self.bias = np.random.uniform(size=self.dim) 
        elif self.init_type == "normal":
            self.weights = np.random.normal(size=self.weights_dim)
            self.bias = np.random.normal(size=self.dim)   
        else:
            self.weights = np.random.standard_normal(size=self.weights_dim) * np.sqrt(2/np.sum(np.array(self.weights_dim)))
            self.bias = np.random.standard_normal(size=self.dim) * np.sqrt(2/np.sum(np.array(self.dim)))          

    def compute(self,prev_input):
        #forward propogation in the layer
        assert self.input_dim == prev_input.shape, "Input is of Wrong Dimension"
        self.prev_input = prev_input
        self.output = self.prev_input @ self.weights + self.bias
        assert self.output.shape == self.dim , "Multiplication Error , Output Dimensions don't match"
        self.output_out = Activation.activate(self.output,self.activation)
        return self.output_out
        

    def compute_gradient(self,next_gradient):
        #backpropogation gradient calculation wrt Error
        o_prime =  Activation.derivative(self.output,self.activation)
        wo_prime = self.prev_input
        next_product = o_prime * next_gradient.reshape(self.dim) 
        self.gradient = wo_prime.T * next_product
        self.bias_gradient =  next_product
        #print("weights_prime : {}\n bias_gradient : {}".format(self.gradient,self.bias_gradient))
        assert self.gradient.shape == self.weights.shape or self.bias_gradient.shape == self.bias.shape, "Gradient dimensions don't match, the tensors"
        next_gradient = self.weights @ next_product.T
        assert next_gradient.shape == (self.prev_input).T.shape, "Activation tensor shape does not match its gradient shape"
        return next_gradient

    def update_weights(self,learning_rate):
        #fix weights and bias
        self.weights = self.weights - learning_rate * self.gradient
        self.bias = self.bias - learning_rate * self.bias_gradient
        #print("weights  : ",self.weights)

class Output(Layer):
    def __init__(self, id, input_dim, dim, init_type='uniform', type='Output', activation='softmax'):
        super().__init__(id, input_dim, dim, init_type=init_type, type=type, activation=activation)

    def compute(self,prev_input):
        self.output = prev_input
        self.output_out = Activation.activate(prev_input,self.activation)
        return self.output_out

    def compute_gradient(self, next_gradient):
        return Activation.derivative(self.output,self.activation) * next_gradient 