import numpy as np 
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.cluster import KMeans
import pandas as pd 
import warnings
warnings.simplefilter(action='error', category=FutureWarning)
pd.set_option('display.max_rows',None,'display.max_columns',None)

text_digit_vals = {}
def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x+=1

			df[column] = list(map(convert_to_int, df[column]))

	return df

df = pd.read_csv('Training.csv')
df.drop(['serial_number','phone_type', 
		 'days_passed','call_duration','previous_contact','date'],1,inplace=True)
print(df.columns)
outcome_contents = df['outcome'].tolist()
for i in range(len(outcome_contents)):
	if outcome_contents[i] == 'no':
		outcome_contents[i] = 0
	if outcome_contents[i] == 'yes':
		outcome_contents[i] = 1
df['outcome'] = outcome_contents
# ---------------------------------
for i in range(len(df)):
	temp = df.iloc[i].tolist()
	if temp.count('unknown') > 3:
		df.drop(i, axis=0)

# ---------------------------------
df = handle_non_numerical_data(df)

X = np.array(df.drop(['outcome'],1))
# quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
# X = quantile_transformer.fit_transform(X)
y = np.array(df['outcome'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
# clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf = svm.SVC(gamma='auto',kernel='rbf')
clf.fit(X_train, y_train)
# accuracy = clf.score(X_test, y_test)
# print(accuracy)




test_df = pd.read_csv('Test.csv')
fin_df = test_df[['serial_number']]
test_df.drop(['serial_number','phone_type', 
		 'days_passed','call_duration','previous_contact','date'],1,inplace=True)
test_df = handle_non_numerical_data(test_df)
X = np.array(test_df)
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
X = quantile_transformer.fit_transform(X)
prediction = clf.predict(X)
print(set(prediction))
# for i in range(len(prediction)):
# 	if prediction[i] == 0:
# 		prediction[i] = 'no' 
# 	if prediction[i] == 1:
# 		prediction[i] = 'yes' 

# fin_df['outcome'] = prediction
# fin_df.to_csv('test_Kneig_2.csv',index=False)
# print(fin_df.head())





