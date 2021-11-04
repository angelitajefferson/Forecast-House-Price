Forecast house price by
Angelita Jefferson


Note: 
* All technical analysis was done in Jupyter notebook (File name: “OpenMe.ipynb”)
* The dataset is in CSV file (File name: “ashfield.csv”)
* The video presentation is “WatchMe.mp4”
* Please refer to the notebook or the presentation video for below analysis steps.

Detail Steps of Technical Analysis for Forecasting House Price:

1. Import basic function and libraries to processing data
	from sklearn.linear_model import LinearRegression
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns

2. Read the dataset file into the pandas dataframe
	ash_raw = pd.read_csv("ashfield.csv")

3. Describe / Inspect the data
	ash_raw.info()

4. Check data quality
	ash = print(ash_raw.duplicated().value_counts())

5. Clean the data, remove duplicate data
	ash = ash_raw.drop_duplicates()

6. Start exploring the data, begin with features selection using linear regression

7. Building linear regression model with all independent variables in the dataset:
	x = S	QFT, Bedrooms, Baths, Age, Occupancy, Pool, Style, Fireplace, Waterfront, DOM
	y = Price

8. Converting the variables from panda to numpy
	y = ash["Price"].to_numpy()
	x = ash[["SQFT","Bedrooms","Baths","Age","Occupancy","Pool","Style","Fireplace","Waterfront","DOM"]].to_numpy()

9. Since the linear regression function already imported at the beginning, I directly create the model and give the model some data to fit
	linear_reg = LinearRegression() 
	I reshape the x’s arrays, ask NumPy to workout the number of rows with 10 columns: 
	linear_reg.fit(x.reshape(-1,10),y)

10. To minimise the squared prediction error, I have to find the values of beta0, beta1, beta2, …., beta10.
	I estimate the parameters using the formula beta_hat formula (please look at the notebook, part 2.1).

11. Estimate parameters (Please refer to the notebook, part 2.1):
	11.1. To calculate the intercept (beta0), I create an array filled with 1’s 
			ones = np.ones((rows,1))
	11.2. Stack the new column of ones horizontally to matrix X
			x = np.hstack((ones,x))
	11.3. Start calculating the betas with matrix transpose,  matrix inverse, and matrix product functions
			betas = np.linalg.inv(x.T @ x) @ x.T @ y
	11.4. Create a loop to calculate the beta continuously until beta10
			print("beta {}: {:.2f}".format(i,betas[i]))
		
12. Full Model to forecast the house price 
	𝑦 = - 50,298.82 + 85.17 SQFT - 25,033.26 Bedrooms + 40,356.08 Baths - 440.44 Age + 6,953.95 Occupancy - 1,792.86 Pool + 1,060.55 Style - 3,073.57 Fireplace + 56,226.11 Waterfront - 21.18 DOM + 𝜀

13. Full Model Evaluation 
	I break the data into 2 set: (1) Training set = To fit the model, (2) Test set = Unseen samples to evaluate the full model.
	
	Then, the full model evaluation steps are as follows:
	13.1 Import the function to split the dataset
			from sklearn.model_selection import train_test_split
	13.2 Import the function to calculate the MSE (The average squared difference between predicted value and true value)
			from sklearn.metrics import mean_squared_error as mse
	13.3 Splitting the dataset, with test size of 40% of the data and random state = 1
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 1)
	13.4 Build the regression model
			linear_reg = LinearRegression()
			linear_reg.fit(x_train,y_train)
	13.4 Predict on the test data (after trained with training dataset, and predict using the unseen samples which is the test dataset)
			y_pred = linear_reg.predict(x_test)
	13.5 Calculate the full model MSE
			test_mse = mse(y_pred,y_test)
			print("MSE Full Model: {:.2f}".format(test_mse))	
	I also evaluate the full model using the regression table, details are as follows:
	13.6 Import the statistic function 
			import statsmodels.api as sm
	13.7 Describe the model and fit the model, then print the regression summary result
			mod = sm.OLS(y_train, x_train)
			res = mod.fit()
			print(res.summary())

Conclusion: the full model has high Mean Squared Error. Hence, to reduce the MSE and produce more accurate model, I reduce the model by excluding independent variables that are the most insignificant. In this case, the independent variables that have p-value > 0.05 are removed. The new reduced model, I only includes: SQFT, Bedrooms, Baths, Age as the independent variables (xnew)

14. I estimate the parameters for the reduced model as the same as I did in the full model, the difference is only at the independent variables. (Please refer to the notebook part 2.2)
	# 'Price' is the dependent variable
	y = ash["Price"].to_numpy()

	# These are independent variables that affect the house price
	xnew = ash[["SQFT","Bedrooms","Baths","Age"]].to_numpy()

	rows = xnew.shape[0] # 1067 rows

	# Create column of ones and stack it horizontally to calculate the intercept beta0
	ones = np.ones((rows,1))
	xnew = np.hstack((ones,xnew))

	# Estimating parameters
	betas = np.linalg.inv(xnew.T @ xnew) @ xnew.T @ y

	for i in range (5):
    	print("beta {}: {:.2f}".format(i,betas[i]))
	
15. Reduced Model to forecast the house price 
	𝑦 = - 30,708.32 + 87.32 SQFT - 28,302.06 Bedrooms + 42,004.71 Baths - 545.71 Age + 𝜀

13. Reduced Model Evaluation 
	I also break the data into 2 set: (1) Training set = To fit the model, (2) Test set = Unseen samples to evaluate the full model.
	I evaluate the reduced model the same way as I did for the full model, the difference is only at the new x training and test data; xnew_test & xnew_train. (Please refer to the notebook part 2.2.1)
		# Train-test split
		xnew_train, xnew_test, y_train, y_test = train_test_split(xnew, y, test_size=0.4, random_state = 1)

		# Build the regression model
		linear_reg = LinearRegression()
		linear_reg.fit(xnew_train,y_train) 

		# Predict on the test data
		y_pred = linear_reg.predict(xnew_test)

		# Calculate the MSE
		test_mse = mse(y_pred,y_test)
		print("MSE Reduced Model: {:.2f}".format(test_mse))

		# Regression results for reduced model
		mod_new = sm.OLS(y_train, xnew_train)
		res_new = mod_new.fit()
		print(res_new.summary())

Conclusion: The MSE for reduced model is lower than the full model and all the independent variables in reduced model have p-value < 0.05 and there is no 0 included in the confidence interval. Hence, there is sufficient evidence to show that reduced model is better.

14. Data Visualisation for 4 significant variables
	
	14.1 SQFT vs House Price (Please refer to the notebook part 3.1)
		The scatter plot is used to recognise a pattern and outliers for this variable.
		Since the matplotlib function already imported in the beginning, I directly create the scatter plot. 
		I also create the slope to draw a clearer relationship between SQFT and house price. 
		After I create a scatter plot, the outliers could be seen clearly, where there are some data that has more than 6300 SQFT. 
		Thus, I create a different colors for the outliers as red.
		The details of the steps are as follows:
			plt.scatter(ash["SQFT"], ash["Price"], alpha = 0.5)
			outlier = ash.query("SQFT > 6300")
			plt.scatter(outlier["SQFT"], outlier["Price"], alpha = 0.5, c = 'red')

			m,b = np.polyfit(ash["SQFT"], ash["Price"],1)
			plt.plot(ash["SQFT"], m * ash["SQFT"] + b, c = 'black')

			plt.xlabel("SQFT")
			plt.ylabel("House Price")
			plt.title("SQFT vs House Price")
			plt.show()

	14.2 Bedrooms vs House Price (Please refer to the notebook part 3.2)
		In reality, the more bedroom the unit has, the higher the unit price would be.
		However, the model indicates that there is a negative linear relationship between bedroom and house price.
		To analyse in more detail, I create a bar plot and a table that show the number of bedrooms and the average house price.
		The dataset that I explored turns out only has 1 sample for each 6,7,8 bedrooms.
		As seen on the table and the bar plot (please refer to the notebook part 3.2), the average price of the 6,7,8 bedrooms are lower than the 5 bedrooms' price.
		This might be possible if the unit does not have any other benefits, such as extremely far from the CBD, overage, huge renovation cost needed, etc.
		This variable is not excluded from the model, since bedroom plays significant factor in reality, although the dataset only has small size for larger bedrooms.
		The details of the steps are as follows:
			avg_bed = ash.groupby("Bedrooms").Price.mean()

			xbed = [1,2,3,4,5,6,7,8]
			plt.bar(xbed, avg_bed)
			plt.xlabel("Bedrooms")
			plt.ylabel("Average House Price")
			plt.title("Bedrooms vs Average House Price")
			plt.show()

			bedrooms = ash.groupby(["Bedrooms"]).Price.agg([np.size, np.mean])
			print(bedrooms)

	14.3 Baths vs House Price (Please refer to the notebook part 3.3)
		The box plot is used to visualise the baths and house price.
		From the box plot, it could be seen that bath and house price also have a positive linear relationship.
		The more baths a unit has, which could be mean that the house is spacious and has more rooms, then the house price would be higher.
		I also create a table which shows the minimum value, maximum value, size per baths, and the average house price per baths.
		The details of the steps are as follows:
			sns.set_style("whitegrid")
			tips = sns.load_dataset("tips")
			ax = sns.boxplot(ash["Baths"], ash["Price"], data=tips)
			plt.title("Baths vs House Price")
			plt.show()

			baths = ash.groupby(["Baths"]).Price.agg([np.min, np.max, np.size, np.mean])
			print(baths)
	14.4 Age vs House Price (Please refer to the notebook part 3.4)
		In visualising between age and house price, I create a 6 bins table and a histogram to show a comprehensive information.
		As seen on the histogram, the age and house price have a negative linear relationship. 
		The older the house, the lower the price, and vice versa. 
		It could also be seen from the bins table, that there are not many old house units are marketed (60 - 80 years old house).
		The details of the steps are as follows:
			plt.hist(ash["Age"], bins = 6)
			plt.xlabel("Age")
			plt.ylabel("House Price")
			plt.title("Age vs House Price")
			plt.show()

			print("\033[1m" + "Age Bins" + "\033[0;0m")
			pd.cut(ash["Age"], bins = 6).value_counts().sort_index()

15. Alternative models
	Besides linear regression model, I use random forest regressor and decision tree regressor to forecast the house price.

	15.1 Linear regression (Please refer to the notebook part 4.1)
		In this step, I create a new data frame called linreg_model_data, which has 4 independent variables (the reduced model) that predict the house price using a linear regression model with a test dataset. The steps are as follows:
		1. I choose the variables and convert them to list types
			xnew_test_SQFT = xnew_test[:,1:2].tolist()
			xnew_test_Bedrooms = xnew_test[:,2:3].tolist()
			xnew_test_Baths = xnew_test[:,3:4].tolist()
			xnew_test_Age = xnew_test[:,4:5].tolist()
		2. Create the new data frame
			linreg_model_data = pd.DataFrame({"SQFT": xnew_test_SQFT, 
                                  "Bedrooms": xnew_test_Bedrooms, 
                                  "Baths": xnew_test_Baths, 
                                  "Age": xnew_test_Age, 
                                  "Linear_Reg_Price": y_pred}) 
		3. Extract all element from each variables
			linreg_model_data['SQFT'] = linreg_model_data['SQFT'].str.get(0)
			linreg_model_data['Bedrooms'] = linreg_model_data['Bedrooms'].str.get(0)
			linreg_model_data['Baths'] = linreg_model_data['Baths'].str.get(0)
			linreg_model_data['Age'] = linreg_model_data['Age'].str.get(0)
		4. Print the new data frame
			print(str(linreg_model_data))

	15.2 Random Forest Regressor (Please refer to the notebook part 4.2)
		To compare the model, in this part I use random forest regressor model. The steps are as follows:
		1. Import function
			from sklearn.ensemble import RandomForestRegressor
		2. Build the model
			forest_model = RandomForestRegressor(random_state = 1)
		3. Give the model some data to fit
			forest_model.fit(xnew_train, y_train)
		4. Predict using the test data
			forest_model_predictions = forest_model.predict(xnew_test)
		5.  Create the new data frame
			forest_model_data = pd.DataFrame({"SQFT": xnew_test_SQFT, 
                                  "Bedrooms": xnew_test_Bedrooms, 
                                  "Baths": xnew_test_Baths, 
                                  "Age": xnew_test_Age, 
                                  "Forest_Mod_Price": forest_model_predictions}) 
		6. Extract all element from each variables
			forest_model_data['SQFT'] = forest_model_data['SQFT'].str.get(0)
			forest_model_data['Bedrooms'] = forest_model_data['Bedrooms'].str.get(0)
			forest_model_data['Baths'] = forest_model_data['Baths'].str.get(0)
			forest_model_data['Age'] = forest_model_data['Age'].str.get(0)		7. Print the new data frame
			print(str(forest_model_data) + "\n")


	15.3 Decision Tree Regressor (Please refer to the notebook part 4.3)
		The steps are as follows:
		1. Import function
			from sklearn.tree import DecisionTreeRegressor
		2. Build the model
			tree_model = DecisionTreeRegressor(random_state = 1)
		3. Give the model some data to fit
			tree_model.fit(xnew_train, y_train)
		4. Predict using the test data
			tree_model_predictions = tree_model.predict(xnew_test)
		5.  Create the new data frame
			tree_model_data = pd.DataFrame({"SQFT": xnew_test_SQFT, 
                                "Bedrooms": xnew_test_Bedrooms, 
                                "Baths": xnew_test_Baths, 
                                "Age": xnew_test_Age, 
                                "Tree_Mod_Price": tree_model_predictions})
		6. Extract all element from each variables
			tree_model_data['SQFT'] = tree_model_data['SQFT'].str.get(0)
			tree_model_data['Bedrooms'] = tree_model_data['Bedrooms'].str.get(0)
			tree_model_data['Baths'] = tree_model_data['Baths'].str.get(0)
			tree_model_data['Age'] = tree_model_data['Age'].str.get(0)
		7. Print the new data frame
			print(str(tree_model_data) + "\n")

16. Models Comparison (Please refer to the notebook part 4.4)
	After using 3 different models to predict the house price, I would like to choose which one is the best method to forecast.
	Therefore, I query random element for 4 independent variables, where SQFT > 5000 | Bedrooms > 2 | Baths > 2 | Age > 2.
		1. I extract the data for each model that matches with the criteria
			1.1 Linear Regression Model
			print("\n" "\033[1m" + "Linear Regression Model Prediction" + "\033[0;0m")
			print(linreg_model_data.sort_values("SQFT").loc[(linreg_model_data['SQFT'] > 5000) 
                                                & (linreg_model_data['Bedrooms'] > 2) 
                                                & (linreg_model_data['Baths'] > 2) 
                                                & (linreg_model_data['Age'] > 20)])
			1.2 Random Forest Regressor 
			print("\n" "\033[1m" + "Forest Model Prediction" + "\033[0;0m")
			print(forest_model_data.sort_values("SQFT").loc[(forest_model_data['SQFT'] > 5000) 
                                                & (forest_model_data['Bedrooms'] > 2) 
                                                & (forest_model_data['Baths'] > 2) 
                                                & (forest_model_data['Age'] > 20)])
			1.3 Decision Tree Regressor
			print("\n" "\033[1m" + "Decision Tree Model Prediction" + "\033[0;0m")
			print(tree_model_data.sort_values("SQFT").loc[(tree_model_data['SQFT'] > 5000) 
                                              & (tree_model_data['Bedrooms'] > 2) 
                                              & (tree_model_data['Baths'] > 2) 
                                              & (tree_model_data['Age'] > 20)])
		2. I extract the actual data that matches with the SQFT that comes out from the 3 models,
		SQFT = 5216, 5736, 6203, 7249, and extract only the first 5 columns.
			actual_data = ash.query("SQFT == 5736 | SQFT == 5216| SQFT == 7249| SQFT == 6203").sort_values("SQFT")
			print("\n" "\033[1m" + "Actual Data" + "\033[0;0m")
			print(actual_data.iloc[:,:5])
		3. Lastly, I compare the MSE between the 3 models. 
		The result shows that the best model is linear regression, followed by the random forest regressor model, then decision tree regressor model. The steps are as follows:
			# Calculate linreg MSE
			linreg_mse = mse(y_pred,y_test)
			print("MSE Linear Regression Model: {:.2f}".format(linreg_mse))

			# Calculate forest MSE
			forest_mse = mse(forest_model_predictions,y_test)
			print("MSE Random Forest Regressor Model: {:.2f}".format(forest_mse))

			# Calculate tree MSE
			tree_mse = mse(tree_model_predictions,y_test)
			print("MSE Decision Tree Regressor Model: {:.2f}".format(tree_mse))
