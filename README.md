# Rock-or-mine-detection using machine learning
# Rock vs Mine Prediction using Machine Learning with Python

## Overview
This project aims to classify sonar signals as either **rock** or **mine** using machine learning techniques. By analyzing sonar data, we train a model to distinguish between the two classes accurately.

## Dataset
The dataset used for this project is the **Sonar Dataset** from the UCI Machine Learning Repository. It consists of 60 numerical features extracted from sonar signals, with labels indicating whether the object is a rock or a mine.

## Features
The dataset includes:
- **60 sonar frequency-based attributes**
- **Label**: Either `R` (Rock) or `M` (Mine)

## Machine Learning Models Used
The following models were implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Rock-vs-Mine-Prediction.git
   cd Rock-vs-Mine-Prediction
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook or Python script:
   ```sh
   jupyter notebook
   ```
   or
   ```sh
   python rock_mine_prediction.py
   ```

## Usage
1. Load the dataset and preprocess the data.
2. Train different machine learning models.
3. Evaluate the models using accuracy, precision, recall, and F1-score.
4. Make predictions on new sonar data.

## Results
The models were evaluated based on performance metrics, and Random Forest achieved the highest accuracy.

## Dependencies
Ensure you have the following Python libraries installed:
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook (optional)

## Contributing
Feel free to fork this repository, submit issues, or open pull requests to improve the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- UCI Machine Learning Repository for the dataset.
- Scikit-Learn and other open-source libraries.

---
Created by Mathusayini Thayalanesan

