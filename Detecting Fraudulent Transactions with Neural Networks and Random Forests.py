"""
Detecting Fraudulent Transactions with Neural Networks and Random Forests

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from imblearn.over_sampling import SMOTE, RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
import xgboost as xgb


# Load financial data
df = pd.read_csv('PaySim.csv')
df.columns
# some column names could be altered to be more readable
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig','oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

df.head()

# The only type of transactions with fraud are Transfers and Cash Outs
df.loc[df.isFraud == 1].type.drop_duplicates().values
df = df[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
len(df)

# Feature engineering
# detcting fraud by amount transfered
df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')].amount.median()
df.loc[(df.isFraud == 0) & (df.type == 'TRANSFER')].amount.median()

df['Fraud_Heuristic'] = np.where(((df['type'] == 'TRANSFER') &
                                 (df['amount'] > 200000)),1,0)

df['Fraud_Heuristic'].sum()

f1_score(y_true=df['isFraud'],y_pred = df['Fraud_Heuristic'])
cm = confusion_matrix(y_true=df['isFraud'],y_pred = df['Fraud_Heuristic'])

# confustion matrix plot helper funciton
def plot_confusion(cm,target_names,title = 'Confusion Matrix', cmap = None, normalize = True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1-accuracy
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks,target_names, rotation = 45)
        plt.yticks(tick_marks,target_names)
        
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    
    thresh = cm.max()/1.5 if normalize else cm.max()/2
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j,i, "{:0.4f}".format(cm[i,j]),
                     horizontalalignment='center',
                     color='white' if cm[i,j] > thresh else 'black')
        else:
            plt.text(j,i, '{:,}'.format(cm[i,j]),
                     horizontalalignment='center',
                     color='white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion.png')
    plt.show()
    
plot_confusion(cm,['Genuine','Fraud'],normalize = False)

# Detecting fraud by transaction time
df['hour'] = df['step']%24
frauds = []
genuine = []
for i in range(24):
    f=len(df[(df['hour']==i) & (df['isFraud'] == 1)])
    g=len(df[(df['hour']==i) & (df['isFraud'] == 0)])
    frauds.append(f)
    genuine.append(g)

sns.set_style('white')
fig, ax = plt.subplots(figsize = (10,6))
gen = ax.plot(genuine/np.sum(genuine), label = 'Genuine')
fr = ax.plot(frauds/np.sum(frauds), label = 'Fraud', dashes = [5,2])
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow = True)
fig.savefig('time.png')
# Genuine transactions happen primarily between 8am and 8pm while Fraudulent transactions occur consistently durring the day

sns.set_style('white')
fig, ax = plt.subplots(figsize = (10,6))
frgen = ax.plot(np.divide(frauds,np.add(genuine,frauds)), label = 'Percent of Fraudulent Transactions')
plt.xticks(np.arange(24))
legend = ax.legend(loc='upper center', shadow=True)
fig.savefig('time_comp.png')

# Check if any overlap between Transefer and Cash Out Fraud
dfFraudTransfer = df[(df.isFraud == 1) & (df.type == 'TRANSFER')]
dfFraudCashOut = df[(df.isFraud == 1) & (df.type == 'CASH_OUT')]
dfFraudTransfer.nameDest.isin(dfFraudCashOut.nameOrig).any()

dfNotFraud = df[(df.isFraud == 0)]
dfFraud = df[(df.isFraud == 1)]
dfFraudTransfer.loc[dfFraudTransfer.nameDest.isin(dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]

len(dfFraud[(dfFraud.oldBalanceDest == 0) & (dfFraud.newBalanceDest == 0) & (dfFraud.amount)]) / (1.0 * len(dfFraud))
len(dfNotFraud[(dfNotFraud.oldBalanceDest == 0) & (dfNotFraud.newBalanceDest == 0) & (dfNotFraud.amount)]) / (1.0 * len(dfNotFraud))

dfOdd = df[(df.oldBalanceDest == 0) & (df.newBalanceDest == 0) & (df.amount)]
len(dfOdd[df.isFraud == 1])/len(dfOdd)
# 70% of transfers where the old and new balances are 0 are fraudulent this could be due to
# there being fraud prevention system in place or there was insufficient funds for the transfer
# can't check for fraud prevention system but we can check for insufficient funds

len(dfOdd[dfOdd.oldBalanceOrig <= dfOdd.amount])/len(dfOdd)
# 90% of transactions have insufficient fund in their original acccounts which could indicate
# that fraud transfers try to drain all the money from the bank account more offen than regular people

# create dummy variables for modeling
# add 'type_' infront of types data for one hot encoding method
df['type'] = 'type_' + df['type'].astype(str)
dummies = pd.get_dummies(df['type'])
df = pd.concat([df,dummies],axis=1)
del df['type']

# Keras
df = df.drop(['nameOrig', 'nameDest', 'Fraud_Heuristic'], axis = 1)
df['isNight'] = np.where((2<=df['hour']) & (df['hour']<=6), 1,0)
df[df['isNight'] == 1].isFraud.mean() 
df = df.drop(['step','hour'],axis = 1)
df.head()

y_df = df['isFraud']
x_df = df.drop(['isFraud'],axis = 1)
Y = y_df.values
X = x_df.values

# train test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
# validation set
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.1, random_state=42)

# Oversample training data as the number of fraudulent transactions are only 0.3% of all data
print(str(round((Y.mean()*100),2))+'%')
sm = SMOTE(random_state=42) #create Synthetic Minority Over-sampling Technique object
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train) # over sample for model training 


# NN model implimentation
model = Sequential()
model.add(Dense(1,input_dim=9))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-5),metrics=['acc'])
model.fit(X_train_res,Y_train_res, epochs=10, batch_size=256, validation_data=(X_val, Y_val))
y_pred = model.predict(X_test)
y_pred[y_pred < 0.5] = 0
y_pred[y_pred > 0.5] = 1
f1_score(y_pred = y_pred, y_true = Y_test)

cm = confusion_matrix(y_pred=y_pred,y_true = Y_test)
plot_confusion(cm, ['Genuine','Fraud'],normalize = False)

# Deeper NN to see if there is an improvment in the F1 score
model2 = Sequential()
model2.add(Dense(16,input_dim=9))
model2.add(Activation('tanh'))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer=SGD(lr=1e-4), metrics=['acc'])
model2.fit(X_train_res,Y_train_res,epochs=10, batch_size=256, validation_data=(X_val,Y_val))
y_pred2 = model2.predict(X_test)

y_pred2[y_pred2 > 0.5] = 1
y_pred2[y_pred2 < 0.5] = 0

f1_score(y_pred=y_pred2,y_true=Y_test)
cm2 = confusion_matrix(y_pred=y_pred2,y_true = Y_test)
plot_confusion(cm2, ['Genuine','Fraud'],normalize = False)


# Random Forest

rf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
rf.fit(X_train,Y_train)
rf_pred = rf.predict(X_test)
f1_score(y_pred = rf_pred,y_true = Y_test)
rf_cm = confusion_matrix(y_pred = rf_pred,y_true = Y_test)
plot_confusion(rf_cm, ['Genuine','Fraud'],normalize = False)

# XGBOOSTed Random Forest
boost = xgb.XGBClassifier(n_jobs=-1)
boost = boost.fit(X_train,Y_train)
boost_pred = boost.predict(X_test)
f1_score(y_pred=boost_pred,y_true=Y_test)
cm_boost = confusion_matrix(y_pred=boost_pred,y_true=Y_test)
plot_confusion(cm_boost,['Genuine','Fraud'], normalize=False)
