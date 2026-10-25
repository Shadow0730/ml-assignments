# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


# %% [markdown]
# the col names 
# the dataframe csv
# class = lcassification

# %%
cols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df=pd.read_csv("diabetes.csv", header=0, names=cols)
def sigmoid(z):
    sig_val = 1/(1+np.exp(-z))
    return sig_val

def initialize_weights(row_vector):
    w= np.zeros_like(row_vector, dtype=float)
    b=0.0
    return w,b

# %% [markdown]
# or we can use if else 
# 
# df['class']=df['class'].map({'g':0,'h':1})
# 
# df['class']=(df['class']=='g).astype(int)

# %%
# df['class'] = (df['class']=='g').astype(int)

# %% [markdown]
# # compute log-loss

# %%
def logloss(y_t, y_p):
    epsilon = 1e-15
    y_p = np.clip(y_p, epsilon, 1 - epsilon)
    n = y_t.shape[0]
    indi_loss = np.dot(y_t, np.log(y_p)) + np.dot((1 - y_t), np.log(1 - y_p))
    loss = (-1 / n) * np.sum(indi_loss)
    return loss

def gradient_dw(x,y,w,b,alpha,N):
    dw=(x*(y-sigmoid(np.dot(w,x)+b)))-((alpha/N)*w)
    return dw

def gradient_db(x,y,w,b):
     db= y-sigmoid(np.dot(w,x)+b)
     return db

def accuracy(y_t, y_p):
    predictions = (y_p >= 0.5).astype(int)
    return np.mean(predictions == y_t)

# %% [markdown]
# # predict function

# %%
def pred(w,b, X):
    N = len(X)
    predict = []
    for i in range(N):
        z=np.dot(w,X[i])+b
        predict.append(sigmoid(z))
    return np.array(predict)

# %%
df.head()

# %% [markdown]
# # Train, validation, test

# %%
train, valid, test = np.split(df.sample(frac=1), [int(0.7*len(df)), int(0.9*len(df))])
train = pd.DataFrame(train, columns=df.columns)
valid = pd.DataFrame(valid, columns=df.columns)
test = pd.DataFrame(test, columns=df.columns)

# %%
def scale_dataset(dataframe):
    x=dataframe[dataframe.columns[:-1]].values
    y=dataframe[dataframe.columns[-1]].values

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

# %%
train, X_train, y_train = scale_dataset(train)
valid, X_valid, y_valid = scale_dataset(valid)
test, X_test, y_test = scale_dataset(test)

# %%
from tqdm import tqdm

def train(X_train, y_train, X_valid, y_valid, epochs, alpha, eta0):
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    w, b = initialize_weights(X_train[0])
    n = X_train.shape[0]

    for epoch in tqdm(range(epochs)):
        for i in range(0,X_train.shape[0]):
            dw = gradient_dw(X_train[i], y_train[i], w, b, alpha, n)
            db = gradient_db(X_train[i], y_train[i], w, b)
            w += eta0 * dw
            b += eta0 * db

        y_train_p = pred(w, b, X_train)
        y_valid_p = pred(w, b, X_valid)

        train_loss.append(logloss(y_train, y_train_p))
        valid_loss.append(logloss(y_valid, y_valid_p))
        train_accuracy.append(accuracy(y_train, y_train_p))
        valid_accuracy.append(accuracy(y_valid, y_valid_p))

    return w, b, train_loss, valid_loss, train_accuracy, valid_accuracy

alpha = 0.1
eta0 = 0.01
epochs = 50
w, b, train_loss, valid_loss, train_accuracy, valid_accuracy = train(X_train, y_train, X_valid, y_valid, epochs, alpha, eta0)

# %%
iterations = range(1, epochs + 1)

test_probability = pred(w, b, X_test)
test_loss = logloss(y_test, test_probability)
test_accuracy = accuracy(y_test, test_probability)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(iterations, train_loss, label='Training loss')
plt.xlabel('Iteration')
plt.ylabel('Cross-entropy loss')
plt.title('Training Loss vs Iteration')
plt.legend()

plt.subplot(2, 2, 1)
plt.plot(iterations, valid_loss, label='Validation loss', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Cross-entropy loss')
plt.title('Validation Loss vs Iteration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(iterations, train_accuracy, label='Training accuracy', color='green')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Iteration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(iterations, valid_accuracy, label='Validation accuracy', color='red')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs Iteration')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Final test loss: {test_loss:.4f}')
print(f'Final test accuracy: {test_accuracy:.4f}')

# %% [markdown]
# # confusion martix
# 

# %%
test_predictions = (test_probability >= 0.5).astype(int)

confusion_matrix = np.zeros((2, 2), dtype=int)

for actual, predicted in zip(y_test.astype(int), test_predictions):
    confusion_matrix[actual, predicted] += 1

print("Confusion Matrix:")
print(confusion_matrix)


