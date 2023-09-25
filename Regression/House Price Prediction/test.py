#import libraries
import pandas as pd
import numpy as np
#loading the data
train_data=pd.read_csv("train.csv")
train_data.head(3)

train_data.drop(columns=['PoolQC','LotConfig','RoofStyle','Fence','MiscFeature',"Alley",'FireplaceQu','MasVnrType'], inplace=True)

# Splitting the data into train and val set
X_train=train_data.iloc[400:,1:-1]
X_val=train_data.iloc[:400,1:-1]
y_train=train_data.iloc[400:,-1]
y_val=train_data.iloc[:400,-1]

categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(exclude=['object']).columns

print(f"categorical columns {categorical_columns}")
print(f"numerical_columns{numerical_columns}")

# fill na values with most frequent(categorical) and with mean value for numerical value
def fillna_numerical(data,numerical_columns):
    for cols in numerical_columns:
        if cols!= 'GarageYrBlt':
            m=data[cols].mean()
            data[cols].fillna(m,inplace=True)
        else:
            most_frequent_value=data[cols].mode()[0]
            data[cols].fillna(most_frequent_value,inplace=True)

    return data


def fillna_categorical(data,categorical_columns):
    for cols in categorical_columns:
        mfv=data[cols].mode()[0]
        data[cols].fillna(mfv,inplace=True)
    return data

# applying above functions
X_train=fillna_numerical(X_train,numerical_columns)
X_val=fillna_numerical(X_val,numerical_columns)
X_train=fillna_categorical(X_train,categorical_columns)
X_val=fillna_categorical(X_val,categorical_columns)

#label encoding
def label_encoding(df):
    label_mappings = {}
    label_counter = 0
    
    # Iterate through columns
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is categorical
            unique_categories = df[col].unique()
            if col not in label_mappings:
                label_mappings[col] = {}
                for category in unique_categories:
                    label_mappings[col][category] = label_counter
                    label_counter += 1
            df[col] = df[col].map(label_mappings[col])
    
    return df
# apply on train and val set
X_train=label_encoding(X_train)
X_val=label_encoding(X_val)


# adjusting common columns
common_cols=list(set(X_train.columns)& set(X_val.columns))
X_train=X_train[common_cols]
X_val=X_val[common_cols]

#normalize
#only to numerical columns
def normalize(data):
    return (data-data.min())/(data.max()-data.min())

# apply nrmalization
X_train=normalize(X_train)
X_val=normalize(X_val)

# convert df into numpy
X_train=X_train.to_numpy()
X_val=X_val.to_numpy()
y_train=y_train.to_numpy()
y_val=y_val.to_numpy()

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(X_train,y_train)

def rmse(y_pred,y_true):
    e=(y_pred-y_true)**2
    return np.sqrt(e)

val_pred=model.predict(X_val)
val_error=rmse(val_pred,y_val)
print(val_error)