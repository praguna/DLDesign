from Layers import *
from LossFunctions import *
import Activation
import pandas as pd
class Model(object):
    def __init__(self):
        #storing output out_dim and meta layer info 
        self.input_dim = (0,0)
        self.outputFunc = lambda x : x
        self.out_dim = (0,0)
        self.Layers = list()
        self.id = 0
        self.meta_data = pd.DataFrame(columns = ["id","Name","Inputdim","Outputdim","Number_of_Parameters","Activation"])
        self.hyper_parameters = pd.DataFrame(columns=["Learning Rate","Loss Function","Batch Size","steps_per_epoch","Epochs","Optimizer"])
        self.meta_data.set_index("id",inplace=True)

    def addInput(self,dim):
        #set input dim
        self.input_dim = dim
        self.out_dim = dim
        assert dim[0]!=0 and dim[1]!=0 and self.id==0, "Invalid Input"
        layer = Layer(self.id,dim,dim,type="Input")
        self.out_dim = dim
        self.Layers.append(layer)
        self.meta_data.loc[self.id] = self.get_layer_info(layer.meta_info)

    def addDense(self,dim,activation="relu",init_type="normal"):
        #Add a dense layer
        assert type(dim) == int , "Provide a integer layer dimension"
        self.id+=1
        layer = Dense(self.id,self.out_dim,(1,dim),activation=activation,init_type=init_type)
        self.Layers.append(layer)
        assert layer.meta_info["Outputdim"]!=0, "Layer dimension is Zero (Try Changing Input or Output)"
        self.out_dim = layer.meta_info["Outputdim"]
        self.meta_data.loc[self.id] = self.get_layer_info(layer.meta_info)    

    def get_layer_info(self,meta_info):
        #construct a row of meta data
        return [meta_info[x] for x in  self.meta_data.columns]  
         
    def build(self,learning_rate=0.01,loss_function="rms",epochs=1,steps_per_epoch=10,batch_size=1,optimizer="grad"):
        #set all base parameters of the DL model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.epochs=epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.hyper_parameters.loc[0] = [self.learning_rate,self.loss_function,self.batch_size,self.steps_per_epoch,self.epochs,self.optimizer]

    def compile(self):
        #initiaize the values in all layers
        for layer in self.Layers :
            layer.initialize()

    def forward_prop(self,input_sample):
        #computes forward propogation result and loss
        result = input_sample
        for layer in self.Layers: 
            result = layer.compute(result) 
        return result

    def back_prop(self,T,P):
        #corrects weights of the layers  
        curr_gradient = LossFunction.compute_derivative(T,P,self.loss_function).reshape(self.out_dim)
        for layer in reversed(self.Layers):   
             #print(layer.id,layer.type)   
             curr_gradient = layer.compute_gradient(curr_gradient)
        for layer in self.Layers:
            layer.update_weights(self.learning_rate)

    def get_layer(self,id):
        #get layer by id
        return self.Layers[id]

    def addOutput(self, activation:str):
        self.id+=1
        output = Output(self.id,input_dim=self.out_dim,activation=activation,dim="output",type="output")
        self.Layers.append(output)
        self.meta_data.loc[self.id] = self.get_layer_info(output.meta_info) 

    def train(self,train_data,train_truth,val_data,val_truth):
        #trains the model for the given conditions and data
        print(self.epochs)
        train_data = train_data.to_numpy()
        train_truth = train_truth.to_numpy()
        val_data = val_data.to_numpy()
        val_truth = val_truth.to_numpy()
        train_loss = []
        val_loss = []
        for i in range(self.epochs):
            print("epoch : {}\n".format(i+1),"*"*10)
            t_l = []
            for _ in range(self.steps_per_epoch):
                l = []  
                for _ in range(self.batch_size):
                    idx =np.random.randint(train_data.shape[0], size=1)
                    sample = train_data[idx]
                    x=self.forward_prop(sample)
                    t = train_truth[idx]
                    loss = LossFunction.compute_loss(t,x,self.loss_function)
                    l.append(loss)
                    self.back_prop(t,x)
                t_l.append(np.mean(l))
            train_loss.append(np.mean(t_l))    
            print("mean training loss : {}".format(train_loss[-1]))
            v=[]    
            for x,y in zip(val_data,val_truth):
                pred = self.forward_prop(x.reshape(1,7))
                loss = LossFunction.compute_loss(y,pred,self.loss_function)
                v.append(loss)
            val_loss.append(np.mean(v))    
            print("mean validation loss :{}".format(val_loss[-1]))
        return train_loss,val_loss        
    
    def predict(self,input):
        # Make a prediction with a network
	    outputs = self.forward_prop(input.reshape(self.input_dim))
	    return outputs  

    def summary(self):
        #prints model summary
        print("*"*50+"  Model summary   "+"*"*50)
        print(self.meta_data)
        print("Trainable Parameters : ",sum(self.meta_data["Number_of_Parameters"]))
        print("#"*50+"  Hyper Parameters  "+"#"*50)
        print(self.hyper_parameters)