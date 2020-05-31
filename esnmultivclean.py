from numpy.random import seed
seed(1)
import numpy as np
from pyESN import ESN #available at https://github.com/cknd/pyESN
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import time

## Função para avaliar performance do modelo de predição. Calcula o RMSE (root mean squared error) para cada produto ("feature")
def results(data, prediction, trainlen, future, plotfigures=False):
	errlist = list()

	for i in range(n_features):
		error=mse(data[trainlen:trainlen + future, i],prediction[:, i],squared=False) 	#"data" representa os dados verdadeiros e "prediction" representa as estimativas.
		errlist.append(error)
		if plotfigures:
			print(str(error))
			plt.style.use('ggplot')
			plt.figure(figsize=(9, 4))
			plt.subplots_adjust(left=0.06, right=0.99, top=0.94)
			# plt.title(products[i])
			plt.plot(range(0,trainlen+future),data[0:trainlen+future,i], color='#595959', label='Vendas reais')
			plt.plot(range(trainlen,trainlen+future),prediction[:,i], 'r', label='Vendas previstas')
			plt.xlabel('Dias', fontsize=14);
			plt.ylabel('Itens vendidos', fontsize=14);
			plt.legend()
	return errlist, np.mean(errlist) 													#"errlist" é uma lista dos erros para cada produto. "np.mean(errlist)" dá a média de erros entre os produtos; útil para encontrar os melhores parâmetros.

starttimetotal = time.time()					#Medição de tempo

df_mba=pd.read_csv("pivot.csv")					#Importa dados de vendas
products=['Coffee','Bread','Tea','Cake'] 		#Seleciona os quatro produtos mais vendidos
df_mba = df_mba.loc[:,products]

n_features = len(products)						#Quantidade de variáveis para predição.

data = np.array(df_mba).astype('float64')

## Parâmetros iniciais para echo state network (ESN), escolhidos arbitrariamente
n_reservoir=5000
sparsity=0.5
random_state=1
spectral_radius=1.7
noise=0.004216965034286

esn = ESN(n_inputs = n_features,				#Número de entradas (produtos) utilizadas na rede.
          n_outputs = n_features,				#Número de saídas (produtos) utilizadas na rede. Neste caso, são iguais.
          n_reservoir=n_reservoir,
          sparsity=sparsity,
          random_state=1,
          spectral_radius=spectral_radius,
          noise=noise)
##

#Primeira predição:
trainlen = 130 									#Quantidade de pontos (dias) a serem usados para treino
future = 30 									#Quantidade de pontos (dias) a serem previstos.

pred_training0 = esn.fit(np.ones((trainlen,n_features)),data[:trainlen,:trainlen])
prediction0 = esn.predict(np.ones((future,n_features)))
errorlist0,errormean0=results(data, prediction0, trainlen, future, plotfigures=False)		#Se desejar ver os erros e plotar gráficos para comparar predições, setar plotfigures = True.
#


#Início de otimização de parâmetros. São quatro variáveis: n_reservoir, sparsity, spectral_radius e noise.
#Primeiramente são buscados o melhor par de spectral_radius e noise, onde melhor significa ter menor RMSE.
#Serão mantidos fixos n_reservoir e sparsity.
n_reservoir=1000								#Atenção: aumentar n_reservoir aumenta muito o tempo computacional.
sparsity=0.3

#Conjunto de spectral_radius linearmente espaçado entre 1 e 1.8:
radiusset=[1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8]


#Conjunto de noise entre 0.0001 e 0.01, igualmente espaçados em escala logarítmica:
noiseset=[0.0001,0.000133352143216,0.000177827941004,0.000237137370566,0.000316227766017,0.000421696503429,0.00056234132519,0.000749894209332,0.001,0.001333521432163,0.001778279410039,0.002371373705662,0.003162277660168,0.004216965034286,0.005623413251904,0.007498942093325,0.01]


meanerrormatrix1=np.zeros((len(radiusset),len(noiseset)))		#Inicializa matriz 2D de erros
debug=1											#Se desejar observar o progresso, manter em 1. Caso contrário, 0.
univindex=-1 									#-1 dá a média de erro de todos os produtos. Valores entre 0..3 dá o erro específico aquele produto. Exemplo: 0 otimiza para 'Coffee', 3 para 'Cake'.

##O loop abaixo monta uma ESN para cada par de spectral_radius e noise
for a in range(0,len(radiusset)):
	for b in range(0, len(noiseset)):
		esn = ESN(n_inputs = n_features,
				  n_outputs = n_features,
				  n_reservoir=n_reservoir,
				  sparsity=sparsity,
				  random_state=1,
				  spectral_radius=radiusset[a],
				  noise=noiseset[b])
		pred_training = esn.fit(np.ones((trainlen,n_features)),data[:trainlen,:trainlen])
		prediction = esn.predict(np.ones((future,n_features)))
		errorlist,errormean=results(data, prediction, trainlen, future, plotfigures=False)
		if debug==1:
			print(a, b, radiusset[a], noiseset[b])
		if univindex==-1: meanerrormatrix1[a,b]=errormean		#Otimiza usando a média dos erros dos 4 produtos
		else: meanerrormatrix1[a,b]=errorlist[univindex]			#Otimiza para o produto anteriormente especificado com "univindex"
##


#Grid search para encontrar o melhor par spectral_radius, noise:
plt.figure(figsize=(10,8))
plt.style.use('default')
im = plt.imshow(meanerrormatrix1.T, vmin=abs(meanerrormatrix1).min(), vmax=abs(meanerrormatrix1).max(), origin='lower',cmap='PuRd')
plt.xticks(np.linspace(0,len(radiusset)-1,len(radiusset)), radiusset, fontsize=12);
noisesetlabels = [ '%.7f' % elem for elem in noiseset ]
plt.yticks(np.linspace(0,len(noiseset)-1, len(noiseset)), noisesetlabels, fontsize=12);
plt.xlabel('spectral_radius', fontsize=14); plt.ylabel('noise', fontsize=14);
cb = plt.colorbar(im);
cb.ax.set_title('RMSE')


minLoss1 = np.min(meanerrormatrix1)
index_min1 = np.where(meanerrormatrix1 == minLoss1)
sr_opt = radiusset[int(index_min1[0])]
noise_opt = noiseset[int(index_min1[1])]
print('O par ótimo é: spectral radius = ', sr_opt,' noise = ',noise_opt,' RMSE = ',minLoss1)
#

#Plot das previsões usando o par ótimo spectral_radius, noise:
spectral_radius=sr_opt
noise=noise_opt

esn = ESN(n_inputs = n_features,
          n_outputs = n_features,
          n_reservoir=n_reservoir,
          sparsity=sparsity,
          random_state=1,
          spectral_radius=spectral_radius,
          noise=noise)

pred_training1 = esn.fit(np.ones((trainlen,n_features)),data[:trainlen,:trainlen])
prediction1 = esn.predict(np.ones((future,n_features)))
errorlist1,errormean1=results(data, prediction1, trainlen, future, plotfigures=True)
#

#Grid search para encontrar o melhor par n_reservoir, sparsity. Como aumenta o n_reservoir, demora bastante:
starttime = time.time()					#Medição de tempo
reservoirset=[3000,4000,5000, 8000]
sparsityset=[0.3,0.35,0.4,0.45,0.5, 0.55]
meanerrormatrix2=np.zeros((len(reservoirset),len(sparsityset)))
debug=1
univindex=-1
for a in range(0,len(reservoirset)):
	for b in range(0, len(sparsityset)):
		esn = ESN(n_inputs = n_features,
				  n_outputs = n_features,
				  n_reservoir=reservoirset[a],
				  sparsity=sparsityset[b],
				  random_state=1,
				  spectral_radius=spectral_radius,
				  noise=noise)
		pred_training = esn.fit(np.ones((trainlen,n_features)),data[:trainlen,:trainlen])
		prediction = esn.predict(np.ones((future,n_features)))
		errorlist,errormean=results(data, prediction, trainlen, future, plotfigures=False)
		if debug==1:
			print(a, b, reservoirset[a], sparsityset[b])
		if univindex==-1: meanerrormatrix2[a,b]=errormean
		else: meanerrormatrix2[a,b]=errorlist[univindex]

plt.style.use('default')
plt.figure(figsize=(9,4))
im = plt.imshow(meanerrormatrix2, vmin=abs(meanerrormatrix2).min(), vmax=abs(meanerrormatrix2).max(), origin='lower',cmap='PuRd')
plt.yticks(np.linspace(0,len(reservoirset)-1,len(reservoirset)), reservoirset);
plt.xticks(np.linspace(0,len(sparsityset)-1, len(sparsityset)), sparsityset);
plt.ylabel('n_reservoir', fontsize=16); plt.xlabel('sparsity', fontsize=16);
cb = plt.colorbar(im);
cb.ax.set_title('RMSE')

minLoss2 = np.min(meanerrormatrix2)
index_min2 = np.where(meanerrormatrix2 == minLoss2)
reserv_opt = reservoirset[int(index_min2[0])]
sparsity_opt = sparsityset[int(index_min2[1])]
print('O par ótimo é: # reservoir = ', reserv_opt,' sparsity = ',sparsity_opt,' RMSE = ',minLoss2)
endtime = time.time()					#Medição de tempo
print(endtime - starttime)				#Medição de tempo
#

#Plot das previsões usando os parâmetros ótimos spectral_radius, noise, n_reservoir, sparsity:
n_reservoir=reserv_opt
sparsity=sparsity_opt
spectral_radius=sr_opt
noise=noise_opt

esn = ESN(n_inputs = n_features,
          n_outputs = n_features,
          n_reservoir=n_reservoir,
          sparsity=sparsity,
          random_state=1,
          spectral_radius=spectral_radius,
          noise=noise)

pred_training2 = esn.fit(np.ones((trainlen,n_features)),data[:trainlen,:trainlen])
prediction2 = esn.predict(np.ones((future,n_features)))
errorlist2,errormean2=results(data, prediction2, trainlen, future, plotfigures=True)
#

endtimetotal = time.time()				#Medição de tempo
print('Tempo total: ' + str(round(endtimetotal - starttimetotal,0)) + 's')	#Medição de tempo

errdf = pd.DataFrame(errorlist2,products)
errdf.rename(columns={0: 'RMSE'})
meanlist = list()
for i in range(n_features):
	meanlist.append(np.mean(data[trainlen:trainlen + future, i]))
errmeanlist = list()
for i in range(n_features):
	errmeanlist.append(errorlist2[i]/np.mean(data[trainlen:trainlen + future, i])*50)
plt.figure(figsize=(9,4))
plt.bar(products,errorlist2, color='#595959', label='RMSE')
plt.xlabel('Produtos', fontsize=14); plt.ylabel('', fontsize=14);
plt.plot(meanlist, label='Média dos 30 dias')
plt.plot(errmeanlist, linestyle='none', marker='o', color='blue', label='RMSE/média x50')
plt.ylim([0,50])
plt.legend()
plt.subplots_adjust(left=0.03, right=1, top=0.97)

