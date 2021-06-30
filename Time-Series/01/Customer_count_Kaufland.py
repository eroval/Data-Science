import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import datetime
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


#Read Data
file = pd.read_excel('Dataset_New.xlsx')
file.set_index("Date", inplace = True)


column_list = ["Average of snow_3h", "Average of snow_1h", "Average of rain_1h", "Average of rain_3h"]
file["Number_of_customers"].replace(0,file["Number_of_customers"].mean(axis=0),inplace=True)
file.update(file[column_list].fillna(0))


#Series
Customers = file["Number_of_customers"]
X = Customers.values
Xtrain = X[:250] # 250 days used for the train data (69%)
Xtest = X[250:] # 115 days used for the test data (31%)
history = [x for x in Xtrain]
predictions = []


# walk-forward validation
for t in range(len(Xtest)):
	model = ARIMA(history, order=(8,1,4))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	predictions.append(output[0])
	history.append(Xtest[t])
    
#Evaluate error rate
error = sqrt(mean_squared_error(Xtest, predictions))
print('Test RMSE: %.3f' % error)
# plot forecasts against actual outcomes
plt.plot(Xtest)
plt.plot(predictions, color='red')