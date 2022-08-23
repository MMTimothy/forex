import requests
import datetime
import pandas as pd
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Activation
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier



def get_rates(base,quote):
	year = datetime.datetime.now().year
	yearlast = year-1
	month = datetime.datetime.now().month
	day = datetime.datetime.now().day
	today = str(year) + "-"+str(month)+"-"+str(5)
	lastyear = str(yearlast) + "-"+str(month)+"-"+str(day)
	url = "https://api.exchangeratesapi.io/history?start_at="+lastyear+"&end_at="+today+"&base="+base
	url1 = "https://api.exchangeratesapi.io/history?start_at=" + lastyear + "&end_at=" + today + "&base=" + quote
	latest = requests.get(url)
	latest1 = requests.get(url1)
	test = latest.json()
	test1 = latest1.json()
	rates = test["rates"]
	rates1 = test1["rates"]
	df = pd.DataFrame(rates)
	df1 = pd.DataFrame(rates1)
	pDf = processDataFrame(df)
	pDf1 = processDataFrame(df1)

	df2= pd.concat([(df.T.sort_index()),(df1.T.sort_index())],axis=1)
	#print(df2)
	#print(df2)
	pDf2 = pd.concat([pDf,pDf1],axis=1)
	#print (pDf2.head(-1))
	quoteProcessed = processQuote(pDf.loc[:,[quote]])
	#print (pDf.loc[:,[quote]])
	#print (quoteProcessed)
	processDatasetKneighbors(pDf2.head(-1),quoteProcessed,pDf2.tail(1))
	processDatasetKneighbors(df2.head(-1),quoteProcessed,df2.tail(1))
	processDatasetForest(pDf2.head(-1),quoteProcessed,pDf2.tail(1))
	processDatasetForest(df2.head(-1),quoteProcessed,df2.tail(1))
	processDatasetGradientBoosted(pDf2.head(-1),quoteProcessed,pDf2.tail(1))
	processDatasetGradientBoosted(df2.head(-1),quoteProcessed,df2.tail(1))
	print (df2.tail(1))
	kerasTest(df2.head(-1),quoteProcessed)
	#kerasTest(pDf2.head(-1),quoteProcessed)

def processDataFrame(df):
	df1 = df.T.sort_index()
	columns = df1.columns
	indexes = df1.index

	#print (df1)
	columnNum = len(df1.columns)
	indexNum = len(df1.index)

	#print (columnNum,indexNum)
	pdFtest = pd.DataFrame()

	for i in range(columnNum):
		for k in range(indexNum):
			if i <= columnNum and k <= indexNum:
				x = df1.loc[indexes[k],columns[i]]
				l = k-1
				y = df1.loc[indexes[l],columns[i]]
				z = x-y
				#print(x,y,z,k,l)
				a = None
				if z > 0:
					a = 1
				elif z == 0:
					a = 0
				elif z < 0:
					a = -1
				pdFtest.loc[indexes[k],columns[i]] = a
	return (pdFtest)

def processQuote(Df):
	dfIndex = Df.index
	dfColumns = Df.columns


	ind = len(dfIndex)
	col = len(dfColumns)

	qPDf = pd.DataFrame()

	for c in range(col):
		for i in range(ind):
			f = i - 1
			if f >= 0:
				qPDf.loc[dfIndex[f],dfColumns[c]] = Df.loc[dfIndex[i],dfColumns[c]]
				# print (i,(i-1))

	return  (qPDf)

def processDatasetKneighbors(inpt,output,pr):
	otp = output.to_numpy()
	ipt = inpt.to_numpy()
	pred = pr.to_numpy()
	otp = otp.flatten('F')
	print (otp.shape)
	print (ipt.shape)

	knn = KNeighborsClassifier(n_neighbors=1)
	X_train,X_test,y_train,y_test = train_test_split(ipt,otp,random_state=0)

	knn.fit(X_train,y_train)

	print("Test set score for Kneighbors is : {:.2f}".format(knn.score(X_test,y_test)))
	print("Train set score for Kneighbors is : {:.2f}".format(knn.score(X_train, y_train)))
	print(knn.predict(pred))

def processDatasetForest(inpt,output,pr):
	otp = output.to_numpy()
	ipt = inpt.to_numpy()
	pred = pr.to_numpy()
	otp = otp.flatten('F')
	print (otp.shape)
	print (ipt.shape)
	
	X_train,X_test,y_train,y_test = train_test_split(ipt,otp,stratify=otp,random_state=0)
	

	forest = RandomForestClassifier(n_estimators=10,random_state=2)
	forest.fit(X_train,y_train)

	print("The test score for forest is : {:.2f}".format(forest.score(X_test,y_test)))
	print("The train score for forest is : {:.2f}".format(forest.score(X_train, y_train)))
	print(forest.predict(pred))
	


def processDatasetGradientBoosted(inpt,output,pr):
	otp = output.to_numpy()
	ipt = inpt.to_numpy()
	pred = pr.to_numpy()
	otp = otp.flatten('F')
	print (otp.shape)
	print (ipt.shape)

	X_train,X_test,y_train,y_test = train_test_split(ipt,otp,stratify=otp,random_state=3)

	gbrt = GradientBoostingClassifier(random_state=0)
	gbrt.fit(X_train,y_train)

	print("The test score for Gradient is : {:.2f}".format(gbrt.score(X_test,y_test)))
	print("The train score for Gradient is : {:.2f}".format(gbrt.score(X_train, y_train)))
	print(gbrt.predict(pred))
	
def kerasTest(X_data,Y_data):
	X_numpy = X_data.to_numpy()
	Y_numpy = Y_data.to_numpy()

	X_train,X_test,y_train,y_test = train_test_split(X_numpy,Y_numpy,random_state=0)
	
	#y_train = np_utils.to_categorical(y_train,10)
	#y_test = np_utils.to_categorical(y_test,10)
	print(X_train.shape)
	model = Sequential()
	model.add(Dense(64,init="uniform",activation="relu",input_dim=len(X_train[0])))
	#model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu",init="uniform"))
	#model.add(Dropout(0.5))
	model.add(Dense(1,activation="sigmoid",init="uniform"))


	sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
	model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

	model.fit(X_train,y_train,epochs=200,batch_size=183)
	
	score =  model.evaluate(X_test,y_test,batch_size=183)
	print(score)
	

	

get_rates("EUR","JPY")

