# # Import required libraries
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta

# class WeatherModelEvaluator:
#     def __init__(self, model, scaler, features):
#         self.model = model
#         self.scaler = scaler
#         self.features = features
        
#     def evaluate_predictions(self, X_test, y_test):
#         """
#         Evaluate model predictions using multiple metrics
#         """
#         # Scale features
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Make predictions
#         y_pred = self.model.predict(X_test_scaled)
        
#         # Calculate metrics
#         metrics = {
#             'MAE': mean_absolute_error(y_test, y_pred),
#             'MSE': mean_squared_error(y_test, y_pred),
#             'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
#             'R2': r2_score(y_test, y_pred)
#         }
        
#         # Calculate percentage of predictions within different error ranges
#         error = np.abs(y_test - y_pred)
#         error_ranges = {
#             '±1°C': np.mean(error <= 1) * 100,
#             '±2°C': np.mean(error <= 2) * 100,
#             '±3°C': np.mean(error <= 3) * 100,
#             '±5°C': np.mean(error <= 5) * 100
#         }
        
#         return metrics, error_ranges, y_pred

#     def plot_prediction_analysis(self, y_test, y_pred, timestamp_test=None):
#         """
#         Create comprehensive visualization of model performance
#         """
#         fig = plt.figure(figsize=(20, 12))
        
#         # 1. Actual vs Predicted Scatter Plot
#         plt.subplot(2, 2, 1)
#         plt.scatter(y_test, y_pred, alpha=0.5)
#         plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#         plt.xlabel('Actual Temperature (°C)')
#         plt.ylabel('Predicted Temperature (°C)')
#         plt.title('Actual vs Predicted Temperature')
        
#         # 2. Error Distribution
#         plt.subplot(2, 2, 2)
#         error = y_pred - y_test
#         sns.histplot(error, kde=True)
#         plt.xlabel('Prediction Error (°C)')
#         plt.ylabel('Count')
#         plt.title('Error Distribution')
        
#         # 3. Error vs Actual Value
#         plt.subplot(2, 2, 3)
#         plt.scatter(y_test, error, alpha=0.5)
#         plt.axhline(y=0, color='r', linestyle='--')
#         plt.xlabel('Actual Temperature (°C)')
#         plt.ylabel('Prediction Error (°C)')
#         plt.title('Error vs Actual Temperature')
        
#         # 4. Time Series Plot (if timestamps are provided)
#         if timestamp_test is not None:
#             plt.subplot(2, 2, 4)
#             plt.plot(timestamp_test, y_test, label='Actual', alpha=0.7)
#             plt.plot(timestamp_test, y_pred, label='Predicted', alpha=0.7)
#             plt.xlabel('Time')
#             plt.ylabel('Temperature (°C)')
#             plt.title('Actual vs Predicted Temperature Over Time')
#             plt.legend()
        
#         plt.tight_layout()
#         plt.show()
        
#     def generate_evaluation_report(self, X_test, y_test, timestamp_test=None):
#         """
#         Generate comprehensive evaluation report
#         """
#         # Get metrics and predictions
#         metrics, error_ranges, y_pred = self.evaluate_predictions(X_test, y_test)
        
#         # Print metrics
#         print("=== Model Performance Metrics ===")
#         print(f"Mean Absolute Error: {metrics['MAE']:.2f}°C")
#         print(f"Root Mean Square Error: {metrics['RMSE']:.2f}°C")
#         print(f"R-squared Score: {metrics['R2']:.4f}")
#         print("\n=== Prediction Accuracy by Range ===")
#         for range_name, accuracy in error_ranges.items():
#             print(f"Predictions within {range_name}: {accuracy:.1f}%")
        
#         # Generate plots
#         self.plot_prediction_analysis(y_test, y_pred, timestamp_test)
        
#         # Calculate additional insights
#         error = np.abs(y_pred - y_test)
        
#         print("\n=== Additional Insights ===")
#         print(f"Maximum Error: {error.max():.2f}°C")
#         print(f"95th Percentile Error: {np.percentile(error, 95):.2f}°C")
#         print(f"Median Error: {np.median(error):.2f}°C")
        
#         return metrics, error_ranges

# def evaluate_model_by_airport(model, scaler, features, X_test, y_test, airports):
#     """
#     Evaluate model performance separately for each airport
#     """
#     results = {}
    
#     for airport in airports:
#         # Filter data for current airport
#         mask = X_test['airport_code_encoded'] == airport
#         X_airport = X_test[mask]
#         y_airport = y_test[mask]
        
#         if len(y_airport) > 0:
#             # Scale features
#             X_airport_scaled = scaler.transform(X_airport)
            
#             # Make predictions
#             y_pred = model.predict(X_airport_scaled)
            
#             # Calculate metrics
#             metrics = {
#                 'MAE': mean_absolute_error(y_airport, y_pred),
#                 'RMSE': np.sqrt(mean_squared_error(y_airport, y_pred)),
#                 'R2': r2_score(y_airport, y_pred)
#             }
            
#             results[airport] = metrics
    
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results).T
#     return results_df

# # Example usage in your main notebook:
# """
# # After training the model:

# # Initialize evaluator
# evaluator = WeatherModelEvaluator(model, scaler, features)

# # Generate evaluation report
# metrics, error_ranges = evaluator.generate_evaluation_report(
#     X_test, 
#     y_test,
#     df_engineered.loc[X_test.index, 'timestamp']  # Pass timestamps if available
# )

# # Evaluate performance by airport
# airport_performance = evaluate_model_by_airport(
#     model,
#     scaler,
#     features,
#     X_test,
#     y_test,
#     df_engineered['airport_code_encoded'].unique()
# )

# print("\n=== Performance by Airport ===")
# print(airport_performance)
# """