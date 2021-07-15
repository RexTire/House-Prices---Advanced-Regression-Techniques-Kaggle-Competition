import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\train.csv')

df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

df['Alley'].fillna('no alley access', inplace=True)

df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0], inplace=True)

df['BsmtQual'].at[948] = 'NB'
df['BsmtQual'].at[332] = 'NB'

df['BsmtCond'].at[948] = 'NB'
df['BsmtCond'].at[332] = 'NB'

df['BsmtExposure'].at[948] = 'NB'
df['BsmtExposure'].at[332] = 'NB'

df['BsmtFinType1'].at[948] = 'NB'
df['BsmtFinType1'].at[332] = 'NB'

df['BsmtFinType2'].at[948] = 'NB'
df['BsmtFinType2'].at[332] = 'NB'

df['BsmtQual'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtCond'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtExposure'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtFinType1'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtFinType2'].fillna('NB', inplace=True) # NB == No Basement

df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

df['FireplaceQu'].fillna('No FirePlace', inplace=True) # NF == No FirePlace

df['GarageType'].fillna('NG', inplace=True) # NG == No Garage
df['GarageYrBlt'].fillna('NG', inplace=True) # NG == No Garage
df['GarageFinish'].fillna('NG', inplace=True) # NG == No Garage
df['GarageQual'].fillna('NG', inplace=True) # NG == No Garage
df['GarageCond'].fillna('NG', inplace=True) # NG == No Garage

df['PoolQC'].fillna('NP', inplace=True) # NP == No Pool

df['Fence'].fillna('NF', inplace=True) # NF == No Fench

df['MiscFeature'].fillna('NMF', inplace=True) # NMF == No MiscFeatures

df['GarageType'].at[346] = '2Attchd'
df['GarageType'].at[1230] = '2BuiltIn'
df['MiscFeature'].at[346] = 'NMF'
df['MiscFeature'].at[1230] = 'NMF'

df.to_csv('cleaned_train_df.csv', inplace=False)
