from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import requests


def get_charts():
	marketPriceUrl = "https://api.blockchain.info/charts/market-price?timespan=30days&rollingAverage=3days&format=json"
	dailyTransactionsUrl = "https://api.blockchain.info/charts/n-transactions?timespan=30days&rollingAverage=3days&format=json"

	marketPrice = requests.get(marketPriceUrl)
	dailyTransactions = requests.get(dailyTransactionsUrl)
	print (dailyTransactions)
	marketPriceJSON = marketPrice.json()
	dailyTransactionsJSON = dailyTransactions.json()
	
	marketPriceValues = marketPriceJSON["values"]
	dailyTransactionsValues = dailyTransactionsJSON["values"]
	print(len(marketPriceValues))
	print(len(dailyTransactionsValues))
	bitcoin = dict()
	bitcoin["marketPrice"] = marketPriceValues
	plot_graph(marketPriceValues,dailyTransactionsValues)
	
def plot_graph(marketPrice,dailyTransactions):
	xPlotMarketPrice = []
	yPlotMarketPrice = []

	xPlotDailyTransactions = []
	yPlotDailyTransactions = []
	
	for pos in range(len(marketPrice)):
		rawX = marketPrice[pos]["x"]
		Xtime = datetime.utcfromtimestamp(rawX).strftime("%d.%m%")
		xPlotMarketPrice.append(Xtime)
		yPlotMarketPrice.append(marketPrice[pos]["y"])
	for pos in range(len(dailyTransactions)):
		rawX = dailyTransactions[pos]["x"]
		Xtime = datetime.utcfromtimestamp(rawX).strftime("%d.%m%")
		xPlotDailyTransactions.append(Xtime)
		yPlotDailyTransactions.append(dailyTransactions[pos]["y"])
	plt.subplot("131")
	plt.plot(yPlotMarketPrice)
	plt.subplot("132")
	plt.plot(yPlotDailyTransactions)
	plt.show()
		
	

get_charts()
	

