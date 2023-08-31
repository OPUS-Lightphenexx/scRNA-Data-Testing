import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import sklearn
from xgboost import plot_importance
from xgboost import plot_tree

def feature_name(file):
    data = pd.read_csv(file)
    return data[0]

def data_loader(file):
    data1 = pd.read_csv(file)
    frame1 = pd.DataFrame(data1)
    frame2 = pd.DataFrame(frame1.values.T)
    frame3 = frame2.drop(0)
    matrix1 = np.array(frame3,dtype='int64')
    return matrix1

matrix1 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E7.75.csv')
matrix2 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E8.25.csv')
matrix3 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E9.25.csv')
matrix4 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E10.5.csv')
matrix5 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E13.5.csv')
matrix6 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/E16.5.csv')
matrix7 = data_loader(r'/home/cavin/桌面/DNABERT-master/Deeplearning-CM/P16.csv')


stack_test = np.vstack((matrix1,matrix2))
stack_test = np.vstack((stack_test,matrix3))
stack_test = np.vstack((stack_test,matrix4))
stack_test = np.vstack((stack_test,matrix5))
stack_test = np.vstack((stack_test,matrix6))
stack_test = np.vstack((stack_test,matrix7))

label = np.array([0]*len(matrix1)+[1]*len(matrix2)+[2]*len(matrix3)
                 +[3]*len(matrix4)+[4]*len(matrix5)+[5]*len(matrix6)
                 +[6]*len(matrix7),dtype='int64')
print(stack_test)
print(label)

x_train, x_test, y_train,y_test = train_test_split(stack_test,label)

model = xgb.XGBClassifier(objective='multi:softmax',
                          num_class=7,
                          eval_metric=['merror','mlogloss'])
model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_train,y_train)])
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0,epochs)


fig,ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis,results['validation_0']['mlogloss'],label='Train')
ax.plot(x_axis,results['validation_1']['mlogloss'],label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.show()

fig,ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis,results['validation_0']['merror'],label='Train')
ax.plot(x_axis,results['validation_1']['merror'],label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.show()

predict = model.predict(x_test)

print(predict)
print(x_test)
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score

print(classification_report(y_test,predict))
print(accuracy_score(y_test,predict))
confusion_matrix = confusion_matrix(y_test,predict)
print(confusion_matrix)
feature_array = model.feature_importances_
np.save('test.npy',feature_array)

print('\n-------------------- Key Metrics --------------------')
print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, predict)))
print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, predict)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, predict, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, predict, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, predict, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, predict, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, predict, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, predict, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, predict, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, predict, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, predict, average='weighted')))

#feature importance

#树
importance = model.feature_importances_.argsort()
plot_importance(model)
plt.show()










