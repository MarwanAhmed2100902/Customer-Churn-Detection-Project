# Customer Churn Prediction

## Overview
This project predicts customer churn using four machine learning models: K-Nearest Neighbors (KNN), Naive Bayes, Support Vector Machine (SVM), and Decision Trees. The dataset undergoes preprocessing, feature engineering, and standardization before model training. The performance of each model is evaluated using various metrics.

## Dataset
The dataset used contains customer details, service usage patterns, and churn labels. Preprocessing steps include handling missing values, encoding categorical variables, and feature scaling.

## Project Structure
- `Customer_Churn_Detection.ipynb` - Jupyter Notebook containing the complete workflow (data loading, preprocessing, model training, evaluation, and comparison).
- `dataset/` - Folder containing the dataset (if publicly available).
- `README.md` - Project documentation.

## Dependencies
To run this project, install the required Python libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Implementation Steps
1. **Data Loading**: Load and inspect the dataset.
2. **Data Preprocessing**: Handle missing values, encode categorical variables, and normalize data.
3. **Exploratory Data Analysis (EDA)**: Visualize data distributions and relationships.
4. **Feature Engineering**: Select relevant features for training.
5. **Oversampling (if applied)**: Address class imbalance using techniques like SMOTE.
6. **Model Training**: Train KNN, Naive Bayes, SVM, and Decision Trees.
7. **Evaluation**: Compare models using accuracy, precision, recall, F1-score, and confusion matrices.
8. **Results Visualization**: Plot model performance metrics for comparison.

## Results and Findings
- The models were evaluated based on multiple performance metrics.
- Spoiler alert: SVM is the winner.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MarwanAhmed2100902/Customer-Churn-Detection-Project.git
   cd Customer-Churn-Detection-Project
   ```
2. Install dependencies (if not installed):
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and run all cells:
   ```bash
   jupyter notebook Customer_Churn_Detection.ipynb
   ```

## Contributing
Feel free to fork the repository and submit pull requests for improvements!

## License
This project is licensed under the MIT License.

