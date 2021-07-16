import pandas as pd

df = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\final_df.csv')
df_test = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\test.csv')

sol = df_test["Id"]

train_df = df.iloc[:1453,:]
test_df = df.iloc[1453:,:]

test_df.drop(["SalePrice"],axis=1,inplace=True)

X_train=train_df.drop(['SalePrice'],axis=1)
y_train=train_df['SalePrice']

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
test_df = scalar.transform(test_df)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)

y_tr=lr.predict(X_train)

y_pred = lr.predict(test_df)

lst = sol
  
lst2 = list(y_pred)
  

df = pd.DataFrame(list(zip(lst, lst2)), columns =['id', 'SalePrice'])

df.to_csv('result.csv',index=False)