import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from jax.scipy.special import logsumexp
import time
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp
import numpy as np
from jax.example_libraries import optimizers
import time
import sys


#----------------------------------------------------------------
def relu(x):
    return jnp.maximum(0, x)

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def shuffle_dataset(X, y, y_onehot, seed):
    indices = jax.random.permutation(seed, jnp.arange(X.shape[0]))
    X_shuffled = jnp.take(X, indices, axis=0)
    y_oh_shuffled = jnp.take(y_onehot, indices, axis=0)
    y_shuffled = y[indices]
    return X_shuffled, y_shuffled, y_oh_shuffled


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = jax.nn.relu(outputs)

    final_w, final_b = params[-1]
    scores = jnp.dot(final_w, activations) + final_b
    probs = jax.nn.softmax(scores)
    return scores, probs

batched_predict = vmap(predict, in_axes=(None, 0))

def accuracy(params, inputs, targets):
    target_class = jnp.argmax(targets, axis=1)
    scores, _ = batched_predict(params, inputs)
    predicted_class = jnp.argmax(scores, axis=1)
    return jnp.mean(predicted_class == target_class)

def NLLloss_categorical(params, inputs, targets):
    scores, preds = batched_predict(params, inputs)
    log_prob = scores - logsumexp(scores, axis=1)[:,None]
    return -jnp.mean(log_prob * targets)

def weight_decay_regularization(params):
    return jax.example_libraries.optimizers.l2_norm(params)

def loss(params, inputs, targets,wd):
    return NLLloss_categorical(params,
                               inputs,
                               targets) +\
            wd * weight_decay_regularization(params)

def NLLlossAT_categorical(params, inputs, targets, eps=0.02):
    """
    Negative log-likelihood loss function with adversarial training
    """
    _, grads = value_and_grad(NLLloss_categorical, argnums=1)(params, inputs, targets)
    loss_ori = NLLloss_categorical(params, inputs, targets)
    loss_ad = NLLloss_categorical(params, inputs + eps*jnp.sign(jnp.array(grads)), targets)
    return (loss_ori + loss_ad)/2

def lossAT(params, inputs, targets,wd):
    return NLLlossAT_categorical(params,
                                inputs,
                                targets) +\
            wd * weight_decay_regularization(params)

@jit
def update(params, x, y, opt_state, wd):
    """ Compute the gradient for a batch and update the parameters """
    #value, grads = value_and_grad(loss)(params, x, y, wd)
    value, grads = value_and_grad(lossAT)(params, x, y, wd)

    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

def logging(params_all, X, y, K, M, log_acc_train, log_nll_train):

    probs = jnp.mean( jnp.array([batched_predict(params_all[i], X)[1] for i in range(M)]), axis=0)

    y_pred = jnp.argmax(probs,axis=1)
    
    y_true = jnp.argmax(y, axis=1)  
    train_acc = jnp.mean(y_pred == y_true)

    y_conf = probs[np.arange(y_true.shape[0]), y_true]
    train_nll = -jnp.mean(jnp.log(y_conf))

    log_acc_train.append(train_acc)
    log_nll_train.append(train_nll)
#---------------------------------------------------------------

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))



#---------------------------------------------------------------------

layer_sizes = [784, 512, 512, 10]
param_scale = 0.1
step_size = 0.01
num_epochs = 5
batch_size = 128
n_targets = 10
wd=1e-3
M = 4
K = layer_sizes[-1]
params_all = [None]*M
opt_state_all = [None]*M

#--------------------------------------------------------------------------

def normalize_data(data):
    """Normalize pixel values to [0, 1] range"""
    return data.astype(jnp.float32) / 255.0

mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_images = normalize_data(train_images)  
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
X = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
X = normalize_data(X) 
y = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)
y_onehot = y

#--------------------
log_acc, log_nll = [], []

opt_init, opt_update, get_params = optimizers.adam(step_size)

for i in range(M):
    # use random seed to randomly initialize the NN
    params_all[i] = init_network_params(layer_sizes, random.PRNGKey(i))
    opt_state_all[i] = opt_init(params_all[i])

key = random.PRNGKey(0)
for epoch in range(num_epochs):
    for i in range(M):
        key, seed = random.split(key)

        X_shuffled, y_shuffled, y_oh_shuffled = shuffle_dataset(X, y, y_onehot, seed)

        params_all[i] = get_params(opt_state_all[i])
        params_all[i], opt_state_all[i], train_loss = update(params_all[i], X_shuffled, y_oh_shuffled, opt_state_all[i], wd)

    logging(params_all, X, y, K, M, log_acc, log_nll)

    print('\r', f'[Epoch {epoch+1}]: Train Acc: {log_acc[-1]:.3f} | Train NLL: {log_nll[-1]:0.3f}', end='')

#---------------------------------------------
def visualize_learning_curves(log_acc, log_nll):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    fig.tight_layout()
    axes[0].plot(log_acc)
    axes[1].plot(log_nll)
    axes[0].title.set_text('Train Acc')
    axes[1].title.set_text('Train NLL')


def visualize_predictions(params_all, X_test, y_test, M=4, num_samples=12):
    """
    Visualize predictions on test samples
    
    Args:
        params_all: List of model parameters for ensemble
        X_test: Test images (flattened)
        y_test: Test labels (one-hot encoded)
        M: Number of models in ensemble
        num_samples: Number of samples to visualize
    """
    # Get ensemble predictions
    probs = jnp.mean(jnp.array([batched_predict(params_all[i], X_test[:num_samples])[1] for i in range(M)]), axis=0)
    
    # Get predictions and true labels
    y_pred = jnp.argmax(probs, axis=1)
    y_true = jnp.argmax(y_test[:num_samples], axis=1)
    
    # Get confidence scores (max probability)
    confidence = jnp.max(probs, axis=1)
    
    # Create subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('MNIST Predictions with Ensemble Confidence', fontsize=16)
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        # Reshape flattened image back to 28x28
        image = X_test[i].reshape(28, 28)
        
        # Plot image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        
        # Create title with prediction info
        pred_label = int(y_pred[i])
        true_label = int(y_true[i])
        conf = float(confidence[i])
        
        # Color code: green for correct, red for incorrect
        color = 'green' if pred_label == true_label else 'red'
        
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {conf:.3f}'
        axes[row, col].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def visualize_uncertainty_distribution(params_all, X_test, y_test, M=4, num_samples=1000):
    """
    Visualize the distribution of prediction uncertainties
    """
    # Get ensemble predictions for a subset
    probs = jnp.mean(jnp.array([batched_predict(params_all[i], X_test[:num_samples])[1] for i in range(M)]), axis=0)
    
    # Calculate entropy (uncertainty metric)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-15), axis=1)  # Predictive entropy
    
    # Get correct/incorrect predictions
    y_pred = jnp.argmax(probs, axis=1)
    y_true = jnp.argmax(y_test[:num_samples], axis=1)
    correct = (y_pred == y_true)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    # Plot entropy distribution only
    ax.hist(entropy[correct], bins=30, alpha=0.7, label='Correct', color='green', density=True)
    ax.hist(entropy[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', density=True)
    ax.set_xlabel('Predictive Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_individual_model_disagreement(params_all, X_test, M=4, num_samples=8):
    """
    Show how individual models in the ensemble disagree on predictions
    """
    # Get predictions from each individual model
    individual_probs = []
    for i in range(M):
        _, probs = batched_predict(params_all[i], X_test[:num_samples])
        individual_probs.append(probs)
    
    individual_probs = jnp.array(individual_probs)  # Shape: (M, num_samples, 10)
    
    # Calculate ensemble prediction
    ensemble_probs = jnp.mean(individual_probs, axis=0)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Individual Model Predictions vs Ensemble', fontsize=16)
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        # Plot image
        image = X_test[i].reshape(28, 28)
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        
        # Get predictions from each model - convert JAX arrays to Python integers
        individual_preds = []
        for m in range(M):
            pred = jnp.argmax(individual_probs[m, i])
            individual_preds.append(int(pred))  # Convert to Python int
        
        ensemble_pred = int(jnp.argmax(ensemble_probs[i]))  # Convert to Python int
        
        # Check if all models agree
        all_agree = len(set(individual_preds)) == 1
        agreement_color = 'green' if all_agree else 'orange'
        
        title = f'Models: {individual_preds}\nEnsemble: {ensemble_pred}'
        axes[row, col].set_title(title, color=agreement_color, fontsize=9)
    
    plt.tight_layout()
    plt.show()

# Add this at the end of your training script, after the training loop:
def add_visualizations_to_script():
    """
    Add these lines at the end of your main script after training is complete
    """
    print("\n" + "="*50)
    print("VISUALIZATION RESULTS")
    print("="*50)

    # 0. Show learning curves
    visualize_learning_curves(log_acc, log_nll)

    # 1. Show individual predictions
    print("Showing prediction samples...")
    visualize_predictions(params_all, X, y_onehot, M, num_samples=12)
    
    # 2. Show uncertainty distributions
    print("Showing uncertainty distributions...")
    visualize_uncertainty_distribution(params_all, X, y_onehot, M, num_samples=1000)
    
    # 3. Show model disagreement
    print("Showing individual model disagreements...")
    visualize_individual_model_disagreement(params_all, X, M, num_samples=8)
    
    # 4. Print some statistics
    ensemble_probs = jnp.mean(jnp.array([batched_predict(params_all[i], X)[1] for i in range(M)]), axis=0)
    y_pred = jnp.argmax(ensemble_probs, axis=1)
    y_true = jnp.argmax(y_onehot, axis=1)
    
    test_accuracy = jnp.mean(y_pred == y_true)
    avg_confidence = jnp.mean(jnp.max(ensemble_probs, axis=1))
    avg_entropy = jnp.mean(-jnp.sum(ensemble_probs * jnp.log(ensemble_probs + 1e-15), axis=1))
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Average Entropy: {avg_entropy:.4f}")

add_visualizations_to_script()