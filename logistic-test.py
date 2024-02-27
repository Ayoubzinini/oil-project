# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

# Load your dataset (replace 'your_data.csv' with your actual dataset file)
# Assuming the dataset has columns 'NIR_spectrum' and 'Acidity'
# Adjust columns accordingly if needed
df = pd.read_excel("data-oil-12+31janv.xlsx",index_col='Unnamed: 0')

# Extract features (NIR spectrum) and target variable (Acidity)
X = df.drop(['Y'],axis=1)
labenc=LabelEncoder()
y = df['Y']
y = labenc.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
best_params={'C': 0.001,'dual': False,'fit_intercept': True,'penalty': 'l2','solver': 'newton-cg'}
model = LogisticRegression(**best_params)


# Train the model on the training set
model.fit(X_train, y_train)

# Predictions on training set
y_train_pred = model.predict(X_train)

# Cross-validation scores
cv_scores = cross_val_score(model, X_train, y_train, cv=3)

# Predictions on test set
y_test_pred = model.predict(X_test)

# Evaluate the model performance
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

r2_cv = np.mean(cv_scores)
rmse_cv = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')))

r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print performance metrics
print(f"R² (Train): {r2_train:.4f}, RMSE (Train): {rmse_train:.4f}")
print(f"R² (Cross Validation): {r2_cv:.4f}, RMSE (Cross Validation): {rmse_cv:.4f}")
print(f"R² (Test): {r2_test:.4f}, RMSE (Test): {rmse_test:.4f}")

# Plot true vs predicted output on test set
plt.scatter(y_test, y_test_pred)
plt.plot(y_test,y_test,'-')
plt.xlabel('True Output')
plt.ylabel('Predicted Output')
plt.title('True vs Predicted Output (Test Set)')
plt.show()

#make it better
better=input("Do you want to make it better : ")
if better=="yes":
    best_r2_test = 0  # Initialize best R² on the test set
    best_selected_features = None  # Initialize best set of selected features
    num_features = X.shape[1]  # Total number of features
    
    while num_features > 0:
        # Initialize Logistic Regression model for feature selection
        model_for_feature_selection = LogisticRegression()
    
        # Initialize RFE
        rfe = RFE(model_for_feature_selection, num_features)
    
        # Fit RFE on the training set
        X_train_selected = rfe.fit_transform(X_train, y_train)
    
        # Get selected features
        selected_features = X.columns[rfe.support_]
    
        # Train a new model with selected features
        model_with_selected_features = LogisticRegression()
        model_with_selected_features.fit(X_train[selected_features], y_train)
    
        # Predictions on test set with selected features
        X_test_selected = X_test[selected_features]
        y_test_pred_selected = model_with_selected_features.predict(X_test_selected)
    
        # Evaluate the model performance with selected features
        r2_test_selected = r2_score(y_test, y_test_pred_selected)
    
        # Check if the current set of selected features gives a better R² on the test set
        if r2_test_selected > best_r2_test:
            best_r2_test = r2_test_selected
            best_selected_features = selected_features
    
        # Reduce the number of features for the next iteration
        num_features -= 1
    
    # Print the best set of selected features and the corresponding R² on the test set
    print("Best Set of Selected Features:", best_selected_features)
    print("Best R² on Test Set:", best_r2_test)
    
    # Train a final model with the best set of selected features
    final_model = LogisticRegression()
    final_model.fit(X_train[best_selected_features], y_train)
    
    # Predictions on test set with the final model
    y_test_pred_final = final_model.predict(X_test[best_selected_features])
    
    # Evaluate the final model performance
    r2_test_final = r2_score(y_test, y_test_pred_final)
    rmse_test_final = np.sqrt(mean_squared_error(y_test, y_test_pred_final))
    
    # Print the final model performance metrics
    print(f"R² (Test with Best Selected Features): {r2_test_final:.4f}, RMSE (Test with Best Selected Features): {rmse_test_final:.4f}")
    
    # Plot true vs predicted output on test set with the final model
    plt.scatter(y_test, y_test_pred_final)
    plt.xlabel('True Output')
    plt.ylabel('Predicted Output (Best Selected Features)')
    plt.title('True vs Predicted Output (Test Set with Best Selected Features)')
    plt.show()

else:
    print("Finnished")