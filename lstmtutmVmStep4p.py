from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
from numpy import array
from numpy import hstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


NUnits = 64
EP = 300
n_steps_in, n_steps_out = 50, 30

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def plot_train_history(history, title):
  loss = history.history['loss']
  epochs = range(len(loss))


  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.title(title)
  plt.legend()
  plt.show()

def AdjXAxisFcast(HistArray, PredIndexStart, YArray, YArrayIndex):
    xaxis = range(PredIndexStart, PredIndexStart+len(YArray[0]))
    yaxis = list()
    for i in range(len(YArray[0])):
        yaxis.append(YArray[0,i,YArrayIndex])
    return xaxis, yaxis

def JustYAxisFcast(HistArray, PredIndexStart, YArray, YArrayIndex):
    for i in range(len(YArray[0])):
        yaxis.append(YArray[0,i,YArrayIndex])
    return xaxis, yaxis

#Begin info import
df_mba = pd.read_csv("pivot.csv")

arrCoffee = df_mba['Coffee'].values
arrBread = df_mba['Bread'].values
arrTea = df_mba['Tea'].values
arrCake = df_mba['Cake'].values

trainsplit = len(df_mba['Date']) - n_steps_in - n_steps_out
if trainsplit + n_steps_in +n_steps_out != len(df_mba['Date']):
    print('Error!')
    exit()


in1 = arrCoffee[:trainsplit]
in2 = arrBread[:trainsplit]
in3 = arrTea[:trainsplit]
in4 = arrCake[:trainsplit]


in1 = in1.reshape((len(in1), 1))
in2 = in2.reshape((len(in2), 1))
in3 = in3.reshape((len(in3), 1))
in4 = in4.reshape((len(in4), 1))


dataset = hstack((in1, in2, in3, in4))
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
n_features = X.shape[2]

model = Sequential()
model.add(LSTM(NUnits, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(NUnits, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse',metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.fit(X, y, epochs=EP, verbose=2)

x_input = hstack((arrCoffee[trainsplit:trainsplit+n_steps_in], arrBread[trainsplit:trainsplit+n_steps_in], arrTea[trainsplit:trainsplit+n_steps_in], arrCake[trainsplit:trainsplit+n_steps_in]))
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=2)

predindex = trainsplit+n_steps_in


xa,ya = AdjXAxisFcast(arrCoffee.tolist(),predindex, yhat,0)
plt.style.use('ggplot')
plt.figure(figsize=(9,4))
plt.subplots_adjust(left=0.06, right=0.99, top=0.94)
# plt.title('Coffee', fontsize=14)
plt.xlabel('Dias', fontsize=14); plt.ylabel('Itens vendidos', fontsize=14);
plt.plot(arrCoffee[:trainsplit+n_steps_in+n_steps_out].tolist(), color = '#595959', label = 'Vendas reais')
plt.plot(xa,ya,'r', label = 'Vendas previstas')
plt.legend()
plt.show()

xa,ya = AdjXAxisFcast(arrBread.tolist(),predindex, yhat,1)
plt.style.use('ggplot')
plt.figure(figsize=(9,4))
plt.subplots_adjust(left=0.06, right=0.99, top=0.94)
# plt.title('Bread')
plt.xlabel('Dias', fontsize=14); plt.ylabel('Itens vendidos', fontsize=14);
plt.plot(arrBread[:trainsplit+n_steps_in+n_steps_out].tolist(), color = '#595959', label = 'Vendas reais')
plt.plot(xa,ya,'r', label = 'Vendas previstas')
plt.legend()
plt.show()

xa,ya = AdjXAxisFcast(arrTea.tolist(),predindex, yhat,2)
plt.style.use('ggplot')
plt.figure(figsize=(9,4))
plt.subplots_adjust(left=0.08, right=0.99, top=0.94)
# plt.title('Tea')
plt.xlabel('Dias', fontsize=14); plt.ylabel('Itens vendidos', fontsize=14);
plt.plot(arrTea[:trainsplit+n_steps_in+n_steps_out].tolist(), color = '#595959', label = 'Vendas reais')
plt.plot(xa,ya,'r', label = 'Vendas previstas')
plt.legend()
plt.show()

xa,ya = AdjXAxisFcast(arrCake.tolist(),predindex, yhat,3)
plt.style.use('ggplot')
plt.figure(figsize=(9,4))
plt.subplots_adjust(left=0.06, right=0.99, top=0.94)
# plt.title('Cake')
plt.xlabel('Dias', fontsize=14); plt.ylabel('Itens vendidos', fontsize=14);
plt.plot(arrCake[:trainsplit+n_steps_in+n_steps_out].tolist(), color = '#595959', label = 'Vendas reais')
plt.plot(xa,ya,'r', label = 'Vendas previstas')
plt.legend()
plt.show()


plot_train_history(model.history,'Loss - Treinamento')