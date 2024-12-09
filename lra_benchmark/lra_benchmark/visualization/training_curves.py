import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """
    Plot the training and validation loss and accuracy curves.
    
    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        train_accs (list): List of training accuracy values.
        val_accs (list): List of validation accuracy values.
        output_dir (str): Directory to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()