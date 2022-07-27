## Optimal Time and Price Analysis

For prediction of optimal time and optimal price of an airline some factors are important, price, stops, duration, optimal hours, dept_hours and arr_time based on the three of the cabins Business, economy and premium economy.

## Understanding the Dataset

The dataset we are working on is a combination of different airlines, their operational days, cabins, dept_city, arrival_city, departure_time, arrival_time, duration, stops, price and dept_flight_time based on which their (optimal_hours and optimal price) as labels are predicted.

**cabin**: The type of service i.e. economy, business class and premium economy

**weekday**: number of weekdays the flight is operational

**Dept_city**: Starting location of flight, source of the flight

**arrival_city**: Ending location of flight, destination of the flight

**departure_time**: Departure time of flight from starting location

**arrival_time**: Arrival time of flight at destination

**duration**: Duration of flight in minutes

**stops**: Number of total stops flight took before landing at the destination.

**Price**: Price of the flight

**Dept_flights_time**: time of the day the flight usually takes place

**optimal_hours**: hours to complete a flight

## Preprocessing and Sentiment Analysis

We split the column of airline into two different columns airline1 and airline2 and removed outliers from our dataset to get good accuracy of our analysis.

Before modelling and splitting we encoded the data using label encoder.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1['Cabin'] = le.fit_transform(df1['Cabin'])
df1['Dept_city'] = le.fit_transform(df1['Dept_city'])
df1['arrival_city'] = le.fit_transform(df1['arrival_city'])
df1['Dept_flights_time'] = le.fit_transform(df1['Dept_flights_time'])
df1['Airline1'] = le.fit_transform(df1['Airline1'])
df1['Airline2'] = le.fit_transform(df1['Airline2'])
df1['departure_time'] = le.fit_transform(df1['departure_time'])

This helps us handle categorical variables, each variable is assigned a unique integer based on alphabetical ordering so that they can be used in prediction of label.

## EDA

**Introduction:**

- dataset comprises of 330938 rows and 15 columns.
- Dataset column variables:
  Price  
  departure_time  
  arrival_time  
  Cabin  
  Dept_city  
  Dept_date  
  arrival_city  
  stops  
  duration  
  weekday  
  dept_hours  
  Dept_flights_time
  optimal_hours  
  Airline1  
  Airline2

**Information of Dataset:**
rcParams['figure.figsize'] = 15.0,8.27
sns.countplot(df1['Dept_flights_time'])
This graph shows the number of flights taken at different times of the day. Morning flights are the greatest followed by afternoon flights, evening flights and the least flights are taken off at night.
sns.countplot(df1['Cabin'])
This shows economy cabin is most preferred one.
**Univariate Analysis:**
Plotted histogram to see the distribution of data for each column and found that few variables are normally distributed. But in order to find relationship between two varibales we need bivariate analysis.
How the Airline variable is related to the Price variable.
Especially for the Airlines like ArIndia and Vistara we can see that the data has a lot of abnormality
\_The variables with skewness > 1 such as wheelbase, compressionratio, horsepower, price are highly positively skewed.
\_The variables with skewness < -1 are highly negatively skewed.
\_The variables with 0.5 < skewness < 1 such as carwidth, curbweight, citympg are moderately positively skewed.
\_The variables with -0.5 < skewness < -1 such as stroke are moderately negatively skewed.
And, the variables with -0.5 < skewness < 0.5 are symmetric i.e normally distributed such as symboling, carheight, boreration, peakrpm, highwaympg.
df1['Price']=np.sqrt(df1['Price'])
Transformed columns with normal distribution using log technique.
**Descriptive Statistics:**

## Model Building:

We created 3 dataframes based on 3 cabins namely business, ecoonomy and premium economy by checking the conditions:
CabinE = df1.loc[df1['Cabin'] == 0]
CabinB = df1.loc[df1['Cabin'] == 1]
CabinPE = df1.loc[df1['Cabin'] == 2]

### Choosing the features

We chose features from the EDA those are related in predicting optimal time and price.

### PCA transformation

We reduced the dimension of dataset to 8 components.
from sklearn.decomposition import PCA
my_pipeline = Pipeline([
('scaler', StandardScaler()),
('DimensionReduction',PCA(n_components=8)),
('model', model)
])
my_pipeline.fit(X_train, y_train)

#### Applying Random Forest Regressor on PCA columns

-Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.
We applied Random Forest Regressor on our pipelines based on cabins, divided those pipelines into three given below:
if(x==0):
file = open('Pipeline For CabinE', 'wb')
pickle.dump(my_pipeline, file)
file.close()
elif(x==1):
file = open('Pipeline For CabinB', 'wb')
pickle.dump(my_pipeline, file)
file.close()
elif(x==2):
file = open('Pipeline For CabinPE', 'wb')
pickle.dump(my_pipeline, file)
file.close()
results.append(my_pipeline.score(X_test, y_test))
x=x+1
and scores we got on prediction are following:
Economy : 0.875104019004475,
Bunsiness : 0.6918831376729949
Premium economy : 0.7095038404235493

## Deployment

you can access our app by following this link []()

### Django

- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We wrote a python code for our application backend using Django and our frontend in html. This application asks user to input dep_city, arrival_city, departure_time, arrival_time and cabin one wants to choose and based on this data optimal hours and price are predicted.

### Heroku

We deployed our Django application to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others.
Airline_and_details.pkl : contains a list of values in which first index indicates the arrival and departure city entered by user on front end. It clusters the data having same arrival and departure city and filter the data whose date is greater than the date provided by the user, to be used in predictions of Optimal time and price.
l = list()
final_list = list()
y = np.array((optimal_price,optimal_time))
for i in Airline_and_details:
if(i[0]==str(key) and i[-4]>day):
x = np.array((i[-1],i[-2]))
l.append((np.linalg.norm(x - y),i))
l.sort(key=lambda i:i[0])
x=0
for i,j in l:
if x<5:
final_list.append(j)
x=x+1
this function finds the closest five pairs (Optimal Price and time using Euclidean distance) with respect to predicted optimal time and price and displays corresponding Airline, Stops, Duration, Departure Time and Arrival Time.
