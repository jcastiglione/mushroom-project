from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

shroom = pd.concat([X, y], axis=1)

shroom.to_csv('data/mushroom.csv', index = False)

#Since all variables are categorical, the only visualization required is this table
#It also includes whether there are missing values, which could be useful.
print(mushroom.variables.drop(columns=["role", "demographic", "description", "units"]))

#Problem 1: Use variables aside from habitat to predict poisonous
#Useful for predicting whether a mushroom is edible when you find it in the wild.
#Models should use only few variables so that the model can be somewhat applicable for memorization.
#However, we don't know which columns to use yet.
df1 = shroom.drop(columns=['habitat'])
df1.to_csv('data/mushroom_poisonous.csv', index=False)

#Problem 2: Using mushroom shape to predict its best farming habitat.
#Only observe edible mushrooms
#Remove color and odor which could be swayed heavily by current growing location.
#While we want to classify, we also want to find mushrooms that could grow better elsewhere.
color_columns = [col for col in shroom.columns if 'color' in col]
df2 = shroom[shroom['poisonous'] == 'e'].drop(columns=['odor'] + color_columns + ['poisonous'])
df2.to_csv('data/mushroom_habitat.csv', index=False)