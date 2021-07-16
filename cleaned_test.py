import pandas as pd

df = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\test.csv')

df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
df['Alley'].fillna('no alley access', inplace=True)

df['Utilities'].fillna(df['Utilities'].mode()[0], inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)

df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)

df['BsmtQual'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtCond'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtExposure'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtFinType1'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtFinType2'].fillna('NB', inplace=True) # NB == No Basement
df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].mean(), inplace=True)
df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].mean(), inplace=True)
df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].mean(), inplace=True)
df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean(), inplace=True)
df['BsmtFullBath'].fillna(df['BsmtFullBath'].mode()[0], inplace=True)
df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mode()[0], inplace=True)

df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
df['FireplaceQu'].fillna('No FirePlace', inplace=True)

df['GarageCars'].fillna(df['GarageCars'].mode()[0], inplace=True)
df['GarageArea'].fillna(df['GarageArea'].mean(), inplace=True)

df['GarageType'].fillna('NG', inplace=True) # NG == No Garage
df['GarageYrBlt'].fillna('NG', inplace=True) # NG == No Garage
df['GarageFinish'].fillna('NG', inplace=True) # NG == No Garage
df['GarageQual'].fillna('NG', inplace=True) # NG == No Garage
df['GarageCond'].fillna('NG', inplace=True) # NG == No Garage

df['GarageYrBlt'].replace('NG', '0000.0', inplace=True)
df['GarageYrBlt'] = df['GarageYrBlt'].astype(float)

df['PoolQC'].fillna('NP', inplace=True) # NP == No Pool
df['Fence'].fillna('NF', inplace=True) # NF == No Fench

df['MiscFeature'].fillna('NMF', inplace=True) # NMF == No MiscFeatures
df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)

df.to_csv('cleaned_test_df.csv', index=False)