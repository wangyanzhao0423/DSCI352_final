# Bitcoin Price Prediction using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict Bitcoin daily high prices using historical time series data.

## Project Structure

```
part2/
├── Problem2.ipynb          # Main Jupyter notebook with LSTM model
├── COINBASE_BTCUSD_1D.csv  # Bitcoin price dataset
└── README.md               # This file
```

## Dependencies Installation

### Option 1: Using pip

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn jupyter
```

### Option 2: Using conda

If you're using conda/miniconda:

```bash
conda install numpy pandas matplotlib scikit-learn jupyter
conda install -c conda-forge tensorflow
```

### Option 3: Create a requirements.txt (Recommended)

Create a `requirements.txt` file with the following content:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
tensorflow>=2.8.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

Then install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Verify Installation

To verify that all packages are installed correctly, run:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
print("All packages installed successfully!")
```

## How to Run the Model

### Prerequisites

1. Ensure all dependencies are installed (see above)
2. Make sure `COINBASE_BTCUSD_1D.csv` is in the same directory as `Problem2.ipynb`

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   
   Or if using JupyterLab:
   ```bash
   jupyter lab
   ```

2. **Open the notebook**:
   - Navigate to the `part2` directory
   - Open `Problem2.ipynb`

3. **Run all cells**:
   - Click `Cell` → `Run All` in the menu
   - Or use the keyboard shortcut: `Shift + Enter` to run cells sequentially

### Notebook Workflow

The notebook follows this sequence:

1. **Data Loading**: Loads Bitcoin price data from CSV
2. **Data Preprocessing**: 
   - Renames 'time' column to 'Date'
   - Converts to datetime and sets as index
   - Keeps only the 'high' price column
   - Splits data into train (2019-2022) and test (2023+)
3. **Data Scaling**: Applies MinMaxScaler to normalize prices (0-1 range)
4. **Sequence Creation**: Creates 30-day sliding windows for LSTM input
5. **Model Building**: Constructs LSTM network with 2 layers (50 units each)
6. **Model Training**: Trains for 20 epochs with Adam optimizer
7. **Prediction**: Makes predictions on test data
8. **Evaluation**: Calculates RMSE and visualizes results

### Expected Runtime

- **Data preprocessing**: ~1-2 seconds
- **Model training**: ~2-5 minutes (depending on hardware)
- **Prediction**: ~10-30 seconds
- **Total**: ~3-6 minutes


## System Requirements

- **Python**: 3.7 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~100MB for dependencies and data

## Notes

- The model uses temporal train/test split (no randomization) to preserve time series structure
- Training data: 2019-2022
- Test data: 2023 onwards
- All prices are scaled to 0-1 range before training
