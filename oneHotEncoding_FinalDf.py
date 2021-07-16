import pandas as pd

df_train = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\cleaned_train_df.csv')
df_test = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\House Prices - Advanced Regression Techniques\cleaned_test_df.csv')

final_df = pd.concat([df_train, df_test], axis=0)

object_features = [features for features in final_df.columns if final_df[features].dtype == 'O']

for f in object_features:
    final_df = pd.get_dummies(final_df, columns=[f])

final_df = final_df.loc[:,~final_df.columns.duplicated()]

final_df.to_csv('final_df.csv', index=False)