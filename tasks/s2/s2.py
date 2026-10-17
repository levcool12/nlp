import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ==================== БАЗОВЫЕ КЛАССЫ СЕТИ ====================

class Layer:
    """Базовый строительный блок нейронной сети."""
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        input_dim = input.shape[1]
        d_layer_d_input = np.eye(input_dim)
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    """Слой с функцией активации ReLU."""
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, np.zeros(input.shape))

    def backward(self, input, grad_output):
        relu_grad_mask = input > 0
        return grad_output * relu_grad_mask


class Dense(Layer):
    """Полносвязный слой: f(x) = W*x + b."""
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_input


# ==================== ФУНКЦИЯ ПОТЕРЬ ====================

def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (-ones_for_answers + softmax) / logits.shape[0]


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С СЕТЬЮ ====================

def forward(network, X):
    activations = []
    inp = X
    activations.append(network[0].forward(inp))
    for i in range(1, len(network)):
        activations.append(network[i].forward(activations[i - 1]))
    assert len(activations) == len(network)
    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    for i in range(len(network) - 1, -1, -1):
        loss_grad = network[i].backward(layer_inputs[i], loss_grad)

    return np.mean(loss)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ==================== СБОРКА И ОБУЧЕНИЕ ПЕРЦЕПТРОНА ====================

np.random.seed(42)

# Архитектура: вход -> Dense -> ReLU -> Dense -> ReLU -> Dense (выход на число классов)
input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(classes_list)

network = []
network.append(Dense(input_dim, hidden_dim))
network.append(ReLU())
network.append(Dense(hidden_dim, 64))
network.append(ReLU())
network.append(Dense(64, output_dim))

# Обучение
train_log = []
val_log = []
n_epochs = 30
batch_size = 64

for epoch in range(n_epochs):
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=batch_size, shuffle=True):
        train(network, x_batch, y_batch)

    train_acc = np.mean(predict(network, X_train) == y_train)
    val_acc = np.mean(predict(network, X_val) == y_val)
    train_log.append(train_acc)
    val_log.append(val_acc)

    print(f"Epoch {epoch + 1}/{n_epochs} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")

# Визуализация обучения
plt.figure(figsize=(8, 5))
plt.plot(train_log, label='train accuracy')
plt.plot(val_log, label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.title('Обучение перцептрона на Word2Vec')
plt.show()

# ==================== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ====================

y_pred = predict(network, X_test)
print("\nОтчёт по тестовой выборке:")
print(classification_report(y_true=y_test, y_pred=y_pred, target_names=classes_list))
print(f"\nИтоговая точность на тесте: {np.mean(y_pred == y_test):.4f}")