# Prediction Of Sale Price For Bulldozer

## Overview

This project aims to predict the sale price of heavy equipment at auction for Fast Iron's "blue book for bulldozers." The dataset is sourced from auction result postings and includes information on usage, equipment types, and configurations.

## Methodology

![Data Science Methodology](https://th.bing.com/th/id/OIP.-OMBZpRNrVYxM0qD5q6wowHaEd?rs=1&pid=ImgDetMain)

### 1. Business Understanding
   - The project addresses the need to value heavy equipment fleets at auctions.

### 2. Analytical Approach
   - Regression modeling is employed to predict sale prices.

### 3. Data Requirements
   - The dataset from Kaggle includes information on usage, equipment types, and configurations.

### 4. Data Collection
   - Data is sourced from Kaggle, with details on sales, machine IDs, models, and more.

### 5. Data Understanding
   - Key columns include SalesID, MachineID, ModelID, datasource, auctioneerID, YearMade, MachineHoursCurrentMeter, and SalePrice.

### 6. Data Preparation
   - Missing values are handled, and irrelevant columns are dropped during data preprocessing.

### 7. Modeling
   - Random Forest regression is chosen for predicting sale prices.

### 8. Evaluation
   - Evaluation is based on the Root Mean Squared Log Error (RMSLE) metric.

### 9. Deployment
   - Plans for model deployment are outlined, including considerations and potential challenges.

### 10. Feedback
   - A feedback loop is established for model performance in real-world scenarios.

## Results

The project results in a regression model that predicts auction sale prices for heavy equipment with a focus on bulldozers.

## Conclusion

The project successfully addresses the business problem, and potential future work includes model enhancements and further feedback-driven improvements.

## Acknowledgments

Thanks to Kaggle for providing the dataset, and to scikit-learn and other Python libraries for their contributions to the project.

## Author

Talal Atef Ahmed

