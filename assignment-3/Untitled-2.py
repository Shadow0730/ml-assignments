# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

data = loadmat("train_32x32.mat")
print(data.keys())

X = data["X"]
y = data["y"]
print(X.shape, y.shape)
X = X.T
X = X.reshape(X.shape[0], -1).astype("float32") / 255.0
y = y.ravel()% 10

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.10, random_state=42
# )


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# 25% of the remaining 80% = 20% of the full dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.22, random_state=42
)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)


# %%
print("X_train shape:", X_train.shape)
print("One sample shape:", X_train[0].shape)
print("Features per sample:", X_train[0].size)

# %%
splits = {
    "Train": y_train,
    "Validation": y_valid,
    "Test": y_test,
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (name, labels) in zip(axes, splits.items()):
    classes, counts = np.unique(labels, return_counts=True)
    ax.bar(classes.astype(str), counts)
    ax.set_title(f"{name} set")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Number of samples")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 5, figsize=(12, 3))

for i, ax in enumerate(axes):
    image = X_train[i].transpose(1, 2, 0)  # (3, 32, 32) -> (32, 32, 3)
    ax.imshow(image)
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
from tensorflow import keras
from tensorflow.keras import layers

nn_model = keras.Sequential([
    layers.Input(shape=(3, 32, 32)),  # one image in your current X layout
    layers.Permute((2, 3, 1)),        # (channels, height, width) -> (height, width, channels)
    layers.Flatten(),                 # 32 × 32 × 3 = 3,072 inputs

    layers.Dense(128, activation="relu"),  # hidden layer 1
    layers.Dense(64, activation="relu"),   # hidden layer 2

    layers.Dense(10, activation="softmax") # output: 10 digit classes
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# %%
history=nn_model.fit(X_train, y_train, epochs=100, 
          batch_size=2000, 
          validation_split=0.2, verbose=0)

# %%
plot_loss(history)
plot_accuracy(history)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    batch_size=150,
    max_iter=15,
    verbose=1
)

param_grid = {
    'hidden_layer_sizes': [(128,64),(300, 300), (500, 500),(600,600)],
    'batch_size': [50,100,150],
    'max_iter': [10, 15, 17]
}

search = GridSearchCV(mlp, param_grid, cv=3, scoring="accuracy", n_jobs=5, verbose=1)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print("Validation accuracy:", search.best_estimator_.score(X_valid, y_valid))

# %%
from sklearn.metrics import log_loss

activations = {
    "sigmoid": "logistic",
    "ReLU": "relu",
    "tanh": "tanh",
    "linear": "identity",
}

best_params=search.best_params_.copy()
epochs=15
classes=np.unique(y_train)
losses={}

for name, activation in activations.items():
    model = MLPClassifier(
        **best_params,
        activation=activation,
        max_iter=1,
        random_state=42
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        train_losses.append(model.loss_)
        val_probabilities = model.predict_proba(X_valid)
        val_losses.append(log_loss(y_valid, val_probabilities, labels=classes))

    losses[name] = (train_losses, val_losses)

# %%
for name, (train_losses, val_losses) in losses.items():
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Training loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation loss")

    plt.title(f"{name} activation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


