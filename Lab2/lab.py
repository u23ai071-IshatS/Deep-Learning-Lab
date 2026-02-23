import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Dataset Preparation
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize to [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Reshape for CNN (28, 28, 1)
    x_train_cnn = x_train.reshape((-1, 28, 28, 1))
    x_test_cnn = x_test.reshape((-1, 28, 28, 1))
    return (x_train_cnn, y_train), (x_test_cnn, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# 2. Model Factory
def build_model(architecture_type='CNN', activation='relu', optimizer_name='adam', 
                use_bn=True, dropout_rate=0.25):
    
    model = models.Sequential()
    
    if architecture_type == 'CNN':
        # Base Architecture as per Lab Manual [cite: 12]
        model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(28, 28, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation=activation))
        model.add(layers.MaxPooling2D((2, 2)))
        if use_bn: model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation=activation))
        model.add(layers.Dense(10, activation='softmax'))
        
    elif architecture_type == 'MLP':
        # Base Architecture for MLP [cite: 15-23]
        model.add(layers.Flatten(input_shape=(28, 28, 1)))
        model.add(layers.Dense(256))
        if use_bn: model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dense(128))
        if use_bn: model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        model.add(layers.Dense(10, activation='softmax'))

    # Optimizer Selection [cite: 32-35]
    if optimizer_name == 'sgd':
        opt = optimizers.SGD(learning_rate=0.01)
    elif optimizer_name == 'momentum':
        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
    else:
        opt = optimizers.Adam()
        
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Execution of Required Tasks
results = []
histories = {}

# Helper to run and log experiments
def run_experiment(exp_name, arch, act, opt, bn, do, epochs=10):
    print(f"Running: {exp_name}...")
    model = build_model(arch, act, opt, bn, do)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
    acc = history.history['val_accuracy'][-1]
    results.append({
        "Experiment": exp_name, "Activation": act, "Optimizer": opt, 
        "BN": bn, "Dropout": do, "Final Accuracy": f"{acc:.4f}"
    })
    return history

# --- Task 1: Activation Function Challenge  ---
histories['Sigmoid'] = run_experiment("Exp 1: Sigmoid", 'CNN', 'sigmoid', 'sgd', True, 0.25)
histories['Tanh'] = run_experiment("Exp 2: Tanh", 'CNN', 'tanh', 'sgd', True, 0.25)
histories['ReLU'] = run_experiment("Exp 3: ReLU", 'CNN', 'relu', 'sgd', True, 0.25)

# --- Task 2: Optimizer Showdown  ---
# (Using ReLU as the best activation from Task 1)
histories['Adam'] = run_experiment("Exp 4: Adam", 'CNN', 'relu', 'adam', True, 0.25)
histories['Momentum'] = run_experiment("Exp 5: Momentum", 'CNN', 'relu', 'momentum', True, 0.25)

# --- Task 3: Normalization & Dropout [cite: 36-39] ---
run_experiment("Exp 6: No BN/No DO", 'MLP', 'relu', 'adam', False, 0.0)
run_experiment("Exp 7: No BN/DO 0.1", 'MLP', 'relu', 'adam', False, 0.1)
run_experiment("Exp 8: With BN/DO 0.25", 'MLP', 'relu', 'adam', True, 0.25)

# 4. Visualizations [cite: 44]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for label, hist in histories.items():
    plt.plot(hist.history['val_accuracy'], label=label)
plt.title('Task 1 & 2: Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for label, hist in histories.items():
    plt.plot(hist.history['loss'], label=label)
plt.title('Task 1 & 2: Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 5. Final Comparison Table [cite: 43]
df_results = pd.DataFrame(results)
print("\n--- Final Comparison Table ---")
print(df_results)