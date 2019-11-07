from Model import *
from Layers import *
import pandas as pd   
import matplotlib.pyplot as plt 
#Dense is a Fully Connected Layer
#Actual Dataset
df = pd.read_csv("wheat_dataset.csv",header=None)
df = df.sample(frac=1).reset_index(drop=True)
train_data = df.head(195)
train_truth = pd.get_dummies(train_data[7])
train_data.drop([7],axis=1,inplace=True)
val_data = df.tail(15)
val_truth = pd.get_dummies(val_data[7])
val_data.drop([7],axis=1,inplace=True)
print(val_data.shape,val_truth.shape,train_data.shape,train_truth.shape)
model = Model()
model.addInput((1,7))
model.addDense(7,activation="relu")
model.addDense(7,activation="relu")
model.addDense(7,activation="sigmoid")
model.addDense(3,activation="relu")
model.addOutput(activation="softmax")
#Set Hyperparameters Default
model.build(loss_function="cross_entropy",learning_rate=0.01,batch_size=32,steps_per_epoch=10,epochs=10)
#description of the model
model.summary()
#initialize layers
model.compile()
#train the model
x,y=model.train(train_data,train_truth,val_data,val_truth)
#plot the graph of errors
fig=plt.figure()
fig.suptitle('test title', fontsize=20)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('ylabel', fontsize=16)
plt.plot(x,label="train_error")
plt.plot(y,label="val_error")
plt.legend(loc='upper right')
plt.show()
#Making predictions
print("Making predictions :")
val_data = val_data.to_numpy()
val_truth =val_truth.to_numpy()
p = model.predict(val_data[0])
t = val_truth[0]
print("truth :{} , prediction :{}".format(t,p))