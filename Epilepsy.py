#%%

import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sklearn
from numpy import random
from scipy import signal
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization, Activation, Flatten, TimeDistributed, AveragePooling1D
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, Input, Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, recall_score,  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.utils import to_categorical
#%%


#Prepare the dataset

main_directory = 'C:/Users/Microsoft/OneDrive/Desktop/Epilepsy/'
foldername = ['Z', 'O', 'N', 'F', 'S']
folders = ['Z/', 'O/', 'N/', 'F/', 'S/']
data = {}

for i in range(5):
    path = main_directory+folders[i]
    dataframe = []
    for j in range(100):
        k = j+1
        if (k<=9 and k>=1):
            strname = '00'
        elif (k>=10 and k<=99):
            strname = '0'
        else:
            strname = ''
        
        filename = path+foldername[i]+strname+str(k)+'.txt'
        
        df = pd.read_csv(filename, sep=" ", header=None, encoding='utf-8')
        dataframe.append(df.T)
    
    data[foldername[i]] = pd.concat(dataframe, ignore_index=True)


#%%
signal = data['S'].iloc[0,:]
plt.figure()
plt.plot(signal)
plt.title('Signal')
plt.xlim(0,1736)
plt.show()
#%%

#Concatanation and make clusters:

#Z-S cluster:    
Cluster1 = pd.concat([data['Z'], data['S']], axis=0)  
LabelSet1 = np.concatenate([np.zeros(data['Z'].shape[0] ), np.ones(data['S'].shape[0])])
    
print(Cluster1.shape)  
print(LabelSet1.shape) 
    
#Convert to dataframe:
LabelSet1_df = pd.DataFrame(LabelSet1, columns=['LabelSet1'])    
    

#%%
 
#O-S cluster:  
    

Cluster2 = pd.concat([data['O'], data['S']], axis=0)  
LabelSet2 = np.concatenate([np.zeros(data['O'].shape[0] ), np.ones(data['S'].shape[0])])
    
print(Cluster2.shape)  
print(LabelSet2.shape) 
    
#Convert to dataframe:
LabelSet2_df = pd.DataFrame(LabelSet2, columns=['LabelSet2'])   


#%%

#ZO-S cluster:  


Cluster3 = pd.concat([data['Z'], data['O'], data['S']], axis=0)  
LabelSet3 = np.concatenate([np.zeros(data['Z'].shape[0] + data['O'].shape[0]), np.ones(data['S'].shape[0])])

    
print(Cluster3.shape)  
print(LabelSet3.shape) 
    
#Convert to dataframe:
LabelSet3_df = pd.DataFrame(LabelSet3, columns=['LabelSet3'])   


#%%

#N-S cluster:  

Cluster4 = pd.concat([data['N'], data['S']], axis=0)  
LabelSet4 = np.concatenate([np.zeros(data['N'].shape[0] ), np.ones(data['S'].shape[0])])
    
print(Cluster4.shape)  
print(LabelSet4.shape) 
    
#Convert to dataframe:
LabelSet4_df = pd.DataFrame(LabelSet4, columns=['LabelSet4'])  



#%%

#F-S  cluster:
    
Cluster5 = pd.concat([data['F'], data['S']], axis=0)  
LabelSet5 = np.concatenate([ np.zeros(data['F'].shape[0] ), np.ones(data['S'].shape[0]) ])
    
print(Cluster5.shape)  
print(LabelSet5.shape) 
    
#Convert to dataframe:
LabelSet5_df = pd.DataFrame(LabelSet5, columns=['LabelSet5'])  



#%%

#NF-S cluster:
    

Cluster6 = pd.concat([data['N'], data['F'], data['S']], axis=0)  
LabelSet6 = np.concatenate([np.zeros(data['N'].shape[0] + data['F'].shape[0]), np.ones(data['S'].shape[0])])

    
print(Cluster6.shape)  
print(LabelSet6.shape) 
    
#Convert to dataframe:
LabelSet6_df = pd.DataFrame(LabelSet6, columns=['LabelSet6']) 


#%%


#ZONF-S

Cluster7 = pd.concat([  data['Z'], data['N'], data['F'], data['S']  ], axis=0)  
LabelSet7 = np.concatenate([ np.zeros( data['Z'].shape[0] + data['N'].shape[0]+ data['F'].shape[0] ), np.ones(data['S'].shape[0])  ])

    
print(Cluster7.shape)  
print(LabelSet7.shape) 
    
#Convert to dataframe:
LabelSet7_df = pd.DataFrame(LabelSet7, columns=['LabelSet7']) 
#%%


#ZO-NFS cluster:
    
Cluster8 = pd.concat([  data['Z'], data['O'], data['N'], data['F'], data['S']  ], axis=0)    
LabelSet8 = np.concatenate([ np.zeros( data['Z'].shape[0] + data['O'].shape[0] ) , np.ones( data['N'].shape[0] + data['F'].shape[0] + data['S'].shape[0]  )    ]) 

    
print(Cluster8.shape)  
print(LabelSet8.shape) 

#Convert to dataframe:
LabelSet8_df = pd.DataFrame(LabelSet8, columns=['LabelSet8']) 


#%%

#ZO-NF cluster:
    

Cluster9 = pd.concat([  data['Z'], data['O'], data['N'], data['F'] ], axis=0)    
LabelSet9 = np.concatenate([ np.zeros( data['Z'].shape[0] + data['O'].shape[0] ) , np.ones( data['N'].shape[0] + data['F'].shape[0]   )    ]) 

    
print(Cluster9.shape)  
print(LabelSet9.shape) 

#Convert to dataframe:
LabelSet9_df = pd.DataFrame(LabelSet9, columns=['LabelSet9']) 

#%%

#ZO-NF-S cluster:
    
label2 = np.full((100,), 2)    
Cluster10 = pd.concat([  data['Z'], data['O'], data['N'], data['F'], data['S'] ], axis=0)    
LabelSet10 = np.concatenate([ np.zeros( data['Z'].shape[0] + data['O'].shape[0] ) , np.ones( data['N'].shape[0] + data['F'].shape[0] ), np.full(data['S'].shape[0], 2)      ]) 

    
print(Cluster10.shape)  
print(LabelSet10.shape) 

#Convert to dataframe:
LabelSet10_df = pd.DataFrame(LabelSet10, columns=['LabelSet10']) 












#%%

#Preaper data for training the model:
    

 # Convert DataFrame to numpy array
Cluster2_array = Cluster2.values
Cluster2_reshaped = Cluster2_array.reshape((Cluster2_array.shape[0], Cluster2_array.shape[1], 1))

#%%
Cluster10_array = Cluster10.values
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))


#%%
#Go to the train the model


    

#%%

#for scatterplot

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted Values- model 5')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  
# plt.show()
#%%

# Define the CNN model as a feature extractor



def create_cnn_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(100, activation='relu')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="cnn_feature_extractor")
    return model 
      
    
# Create the CNN feature extractor

cnn_model = create_cnn_feature_extractor((4097, 1))
cnn_model.compile(optimizer=Adam(learning_rate=1e-8), loss='sparse_categorical_crossentropy')
#Summary of the Model:
cnn_model.summary()  

#%%

#Train the model with SVM

print("This part is training with svm as classifier")
# Reshape input data to (samples, timesteps, features)
Cluster2_array = Cluster2.values  # Assuming Cluster1 is a DataFrame
Cluster2_reshaped = Cluster2_array.reshape((Cluster2_array.shape[0], Cluster2_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster2_reshaped, LabelSet2), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster2_reshaped[train_index], Cluster2_reshaped[test_index]
    y_train, y_test = LabelSet2[train_index], LabelSet2[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=9, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0)
    svm.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_svm = svm.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_svm)
    sensitivity = recall_score(y_test, y_pred_svm, average='binary') 
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_svm))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")



#%%

#Train the model with Random Forest
print("this code is training with rf:")

# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values 
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=7, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=100, min_samples_split=3, random_state=42)
    rf.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_rf = rf.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_rf)
    sensitivity = recall_score(y_test, y_pred_rf, average='binary')  # Adjust 'average' if not binary classification
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_rf))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")


#%%

print("this code is training with knn:")
#Train the model with KNN

#Because in the paper dont mentioned anything about n_neghbours that we must set in first,-
#so I chosee a defult number for this parameter



# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

#10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=7, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train KNN
    n_neighbors = 5 
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_knn = knn.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_knn)
    sensitivity = recall_score(y_test, y_pred_knn, average='binary') 
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_knn))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")



#%%

#%% Train the model with Gaussian NB:

print("this code is training with GaussianNB:")
# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values 
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=7, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_gaussianNb = gnb.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_gaussianNb)
    sensitivity = recall_score(y_test, y_pred_gaussianNb, average='binary')  
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_gaussianNb).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_gaussianNb))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")



#%%

#%% 
#Train the model with Desicion tree

print("this code is training with decision tree:")

# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values  
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=7, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Decision Tree using Gini index
    dt = DecisionTreeClassifier(criterion='gini', random_state=42)  
    
    # Fit the Decision Tree
    dt.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_dt = dt.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_dt)
    sensitivity = recall_score(y_test, y_pred_dt, average='binary')  
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_dt))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")


#%%

#Train the model with Logistic Regression

print("this code is training with logestic regression:")
# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values  
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=1, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_lr = lr.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_lr)
    sensitivity = recall_score(y_test, y_pred_lr, average='binary')  # Adjust 'average' if not binary classification
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_lr))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")


#%%


#Train the model with Adaboost

print("this code is training with Adaboost:")


# Reshape input data to (samples, timesteps, features)
Cluster1_array = Cluster1.values  # Assuming Cluster1 is a DataFrame
Cluster1_reshaped = Cluster1_array.reshape((Cluster1_array.shape[0], Cluster1_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster1_reshaped, LabelSet1), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster1_reshaped[train_index], Cluster1_reshaped[test_index]
    y_train, y_test = LabelSet1[train_index], LabelSet1[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=7, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train AdaBoost
    ada = AdaBoostClassifier(n_estimators=3, algorithm='SAMME', random_state=42)
    ada.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred = ada.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, average='macro')  # For multi-class classification
    
    # Calculate specificity (per class)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%





#For 3 classification- Cluster10:


#%%

    
# Define the CNN model as a feature extractor
def create_cnn_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(100, activation='relu')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="cnn_feature_extractor")
    return model

# Create the CNN feature extractor
cnn_model = create_cnn_feature_extractor((4097, 1))
cnn_model.compile(optimizer=Adam(learning_rate=1e-8), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()


#%%
# Train the model with SVM
print("training model with svm for cluster10:")



# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values  
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=15, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
    svm.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_svm = svm.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_svm)
    sensitivity = recall_score(y_test, y_pred_svm, average='macro') 
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_svm)
    tn = np.diag(cm).sum()
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    specificity = tn / (tn + fp.sum()) if (tn + fp.sum()) > 0 else 0
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_svm))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%
#Train the model with rf:
print("Train cluster10 with Random Forest")


# Train the model with Random Forest
print("This code is training with Random Forest:")

# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values 
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=100, min_samples_split=3, random_state=42)
    rf.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_rf = rf.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_rf)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_rf, average='macro')  # or 'weighted'
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_rf)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_rf))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%

# Train the model with KNN
print("This code is training with KNN:")

# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values 
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5) 
    knn.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_knn = knn.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_knn)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_knn, average='macro') 
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_knn)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_knn))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%

# Train the model with Gaussian NB
print("This code is training with Gaussian NB:")


# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values 
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_gnb = gnb.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_gnb)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_gnb, average='macro')  
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_gnb)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_gnb))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%

# Train the model with Logistic Regression

print("This code is training with Logistic Regression:")

# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values 
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, multi_class='auto', random_state=42)  # Adjust parameters as needed
    log_reg.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_log_reg = log_reg.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_log_reg)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_log_reg, average='macro')  # or 'weighted'
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_log_reg)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_log_reg))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")

#%%


# Train the model with Decision Tree
print("This code is training with Decision Tree:")

# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values  
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_val, y_val), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train Decision Tree
    dt = DecisionTreeClassifier(criterion='gini', random_state=42)  
    dt.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_dt = dt.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_dt)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_dt, average='macro')
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_dt)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_dt))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")
#%%



print("This code is training with AdaBoost:")

# Reshape input data to (samples, timesteps, features)
Cluster10_array = Cluster10.values  
Cluster10_reshaped = Cluster10_array.reshape((Cluster10_array.shape[0], Cluster10_array.shape[1], 1))

# Prepare for 10-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store results
accuracies = []
sensitivities = []
specificities = []

# 10-fold cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(Cluster10_reshaped, LabelSet10), 1):
    print(f"Fold {fold}")
    
    # Split the data
    X_train, X_test = Cluster10_reshaped[train_index], Cluster10_reshaped[test_index]
    y_train, y_test = LabelSet10[train_index], LabelSet10[test_index]
    
    # Ensure labels are in range [0, 1, 2]
    y_train = np.array(y_train) - np.min(y_train)
    y_test = np.array(y_test) - np.min(y_test)

    # Convert labels to categorical format for CNN training
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Further split training data into train and validation (70% for training, 30% for validation)
    X_train, X_val, y_train_cat, y_val_cat = train_test_split(X_train, y_train_cat, test_size=0.3, random_state=42)
    
    # Train the CNN feature extractor
    cnn_model.fit(X_train, y_train_cat, epochs=10, batch_size=10, validation_data=(X_val, y_val_cat), verbose=0)
    
    # Extract features
    train_features = cnn_model.predict(X_train)
    test_features = cnn_model.predict(X_test)
    
    # Train AdaBoost
    ada = AdaBoostClassifier(n_estimators=3, random_state=42)
    ada.fit(train_features, y_train)
    
    # Predict and evaluate
    y_pred_adaboost = ada.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred_adaboost)
    
    # Sensitivity (recall) for multi-class classification
    sensitivity = recall_score(y_test, y_pred_adaboost, average='macro')
    
    # Calculate specificity
    cm = confusion_matrix(y_test, y_pred_adaboost)
    specificity = np.mean([cm[i, i] / (cm[i, i] + np.sum(cm[:, i]) - cm[i, i]) for i in range(cm.shape[0])])
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Sensitivity: {sensitivity:.4f}")
    print(f"Fold {fold} Specificity: {specificity:.4f}")
    print(classification_report(y_test, y_pred_adaboost))

# Print overall results
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
print(f"Average Sensitivity: {np.mean(sensitivities):.4f} (+/- {np.std(sensitivities):.4f})")
print(f"Average Specificity: {np.mean(specificities):.4f} (+/- {np.std(specificities):.4f})")


