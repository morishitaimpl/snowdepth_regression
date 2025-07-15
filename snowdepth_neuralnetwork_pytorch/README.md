# PyTorch Neural Network for Snow Depth Prediction

This directory contains a PyTorch implementation of a neural network for predicting snow depth based on meteorological data.

## Files

- `config.py`: Neural network configuration and model definition
- `train.py`: Training script for the neural network
- `data_2016_2025.csv`: Training dataset with meteorological data
- `predict_snowdepth.py`: Interactive prediction script for trained models

## Model Architecture

The neural network (`neuralnetwork` class in `config.py`) uses:
- Input layer: 13 features (meteorological variables)
- Hidden layers: 64 → 64 neurons with ReLU activation
- Output layer: 1 neuron (snow depth prediction)
- Configuration parameters defined in `config.py`

## Usage

### Training
```bash
python train.py data_2016_2025.csv output_directory
```

### Interactive Prediction
```bash
python predict_snowdepth.py model.pth [output_directory]
```

The prediction script now uses **terminal input** instead of CSV file reading:
1. Load the trained model
2. Prompt user to input meteorological data interactively
3. Generate snow depth prediction
4. Optionally compare with actual value if provided
5. Save results to text file and generate comparison graph

## Recent Changes (2025-01-15)

### Modified predict_snowdepth.py
- **Changed from CSV input to interactive terminal input**
- **Restructured main processing outside of functions** for better readability
- **Integrated config.py as module** for settings (epochSize, batchSize, input_size)
- **Fixed class name consistency** - uses `neuralnetwork` class correctly
- **Simplified evaluation workflow** for single predictions
- **Added interactive actual value comparison** (optional)
- **Enhanced result saving** with detailed text output

### Updated config.py
- **Fixed super() call** from `SnowDepthPredictor` to `neuralnetwork`
- **Added input_size parameter** (13) for model configuration
- **Updated default input_size** in class constructor

### Key Improvements
- **User-friendly terminal interface** - no need for CSV files
- **Real-time prediction capability** - input data and get immediate results
- **Better code organization** - main logic not wrapped in functions
- **Consistent class naming** throughout the codebase
- **Enhanced error handling** for user input validation

## Input Features

The model expects 13 meteorological features (input interactively):
1. month
2. day
3. land_atmosphere
4. sea_atmosphere
5. precipitation
6. temperature
7. humidity
8. wind_speed
9. wind_direction
10. sum_insolation
11. sum_sunlight
12. snow_falling
13. melted_snow

## Output

- Interactive snow depth prediction in centimeters
- Optional evaluation metrics when actual value is provided
- Comparison visualization (bar chart)
- Detailed results saved to text file
