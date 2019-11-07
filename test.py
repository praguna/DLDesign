from Model import *
from Layers import *

#Dummy Data set
train_data = np.array([[0.1,0.2,0.7]])
train_truth = np.array([[1,0,0]])
weights_1 = np.array([[0.1,0.2,0.3],[0.3,0.2,0.7],[0.4,0.3,0.9]])
weights_2 =  np.array([[0.2,0.3,0.5],[0.3,0.5,0.7],[0.6,0.4,0.8]])
weights_3 =  np.array([[0.1,0.4,0.8],[0.3,0.7,0.2],[0.5,0.2,0.9]])
bias = np.zeros(shape=(1,3))

#Define Model
model = Model()
model.addInput((1,3))
model.addDense(3,activation="relu")
model.addDense(3,activation="sigmoid")
model.addDense(3,activation="relu")
model.addOutput(activation = "softmax")
model.compile()
# model.get_layer(1).set_weights(weights_1,bias)
# model.get_layer(2).set_weights(weights_2,bias)
# model.get_layer(3).set_weights(weights_3,bias)
#check forward prop
x=model.forward_prop((train_data).reshape(1,3))
assert np.max(x-[[0.19858, 0.28559, 0.51583]]) < 1e5,"Forward propogation error"
model.build(loss_function="cross_entropy",learning_rate=1)
model.summary()

#check back prop
print("*"*10+" Back Prop Values" +"*"*10)
for i in range(10) : 
    x = model.forward_prop(train_data)
    print("output : {} , expected : {}".format(x,train_truth))
    loss = LossFunction.compute_loss(train_truth,x,"cross_entropy")
    print("loss : {}".format(loss))
    if abs(loss) <= 1e-10 : break 
    model.back_prop(np.array(train_truth),x)

assert np.max(np.abs((x - train_truth))) < 1e-5, "Back propogation error"

P = np.where(np.amax(x) == x)
T = np.where(np.amax(train_truth) == train_truth)
print("prediction : {} , actual : {}".format(P,T))