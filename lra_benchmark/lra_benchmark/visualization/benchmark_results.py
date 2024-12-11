import matplotlib.pyplot as plt
import os

def plot_benchmark_results(metrics, output_dir):
    """
    Plot relevant benchmark results including test accuracy, test loss, model size, and inference time.
    
    Args:
        metrics (dict): The dictionary containing results returned by `run_benchmark`.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot Test Accuracy and Test Loss
    test_accuracy = metrics.get("test_accuracy", None)
    test_loss = metrics.get("test_loss", None)
    
    if test_accuracy is not None and test_loss is not None:
        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.bar(['Test Accuracy'], [test_accuracy * 100], color='green')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.bar(['Test Loss'], [test_loss], color='red')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_metrics.png'))
        plt.close()

    # 2. Plot Model Size and Inference Time
    model_size = metrics.get("model_size", None)
    inference_time = metrics.get("inference_time", None)

    if model_size is not None and inference_time is not None:
        plt.figure(figsize=(6, 4))
        plt.subplot(1, 2, 1)
        plt.bar(['Model Size'], [model_size / 1e6], color='blue')  # In millions of parameters
        plt.ylabel('Model Size (Millions of Parameters)')
        plt.title('Model Size')
        
        plt.subplot(1, 2, 2)
        plt.bar(['Inference Time'], [inference_time], color='orange')
        plt.ylabel('Time (Seconds)')
        plt.title('Inference Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_inference.png'))
        plt.close()

    # 3. Plot Training and Validation Curves (Optional: Requires modifications to track these metrics)
    # Assuming you track training/validation losses and accuracies during training
    # For now, we just include placeholders (you can replace with actual data if tracked during training)
    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    train_accs = metrics.get("train_accs", [])
    val_accs = metrics.get("val_accs", [])
    
    if train_losses and val_losses and train_accs and val_accs:
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 6))
        # Plot Loss Curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot Accuracy Curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label='Training Accuracy', color='green')
        plt.plot(epochs, val_accs, label='Validation Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_validation_curves.png'))
        plt.close()

