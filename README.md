# MLOps House Price Prediction

**Student ID:** 25L9002

## Project Structure


mlops-house-price-25L9002/
├── data/
├── src/
│   └── train_25L9002.py
├── model/
├── .gitignore
├── requirements.txt
└── README.md


## Installation and Execution

Run the following commands from the project root directory.

### 1. Install the required Python packages


python -m pip install -r requirements.txt


### 2. Run the training script


python src/train_25L9002.py


The script loads the house-price dataset, trains the machine-learning model, evaluates its performance, and saves the trained model in the' model/' directory.

### 3. Verify the generated model

After successful training, the trained model should be available as:

text
model/house_price_model_25L9002.joblib

