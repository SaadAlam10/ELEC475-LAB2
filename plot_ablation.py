from utils.visualization import plot_ablation_summary

ablation_results = {
    "1. Base LeNet-5": {"val_accs": [46.67, 50.16, 51.22, 52.95, 54.89]},
    "2. Modernized LeNet": {"val_accs": [53.47, 59.54, 62.79, 64.72, 65.72]},
    "3. AlexNet (No Norm)": {"val_accs": [46.16, 55.68, 59.73, 64.83, 67.31, 69.49, 70.18, 71.83, 72.68, 73.98]},
    "4. AlexNet + LRN": {"val_accs": [45.34, 53.74, 60.01, 64.25, 67.37, 69.56, 71.23, 73.24, 73.91, 73.61]},
    "5. AlexNet + BatchNorm": {"val_accs": [52.03, 61.82, 66.52, 72.05, 73.78, 70.25, 72.04, 78.02, 77.92, 76.91]},
    "6. AlexNet + BN + Aug": {"val_accs": [49.42, 59.42, 63.05, 69.83, 74.34, 72.02, 74.67, 78.13, 78.53, 79.43, 81.32, 79.34, 80.59, 77.18, 80.51]},
    "7. PyTorch Zoo AlexNet": {"val_accs": [38.65, 46.61, 54.01, 57.55, 58.60, 60.00, 62.99, 63.45, 64.44, 63.79]},
}

plot_ablation_summary(ablation_results, save_path="./plots/ablation_summary.png")
