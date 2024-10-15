# Sales Prediction Competition

## Introduction
Our group participated in a coding challenge aimed at building a machine learning model to predict sales for the third year, using patterns from the previous two years of sales data. The challenge required us to consider key factors such as seasonality and trends, which could significantly impact the accuracy of sales predictions.

Our group selected Google Colab as our main development platform to enhance collaboration and efficiency. This platform enabled us to share code and collaborate in real-time, greatly improving our teamwork. With integrated support for popular machine learning libraries such as TensorFlow, scikit-learn, and Pandas, Colab provided the necessary cloud-based computational resources to train and test our models without the limitations of local hardware. Throughout the project, we took advantage of its collaborative features to explore various machine learning techniques and feature engineering strategies, fostering discussions and troubleshooting that allowed us to iteratively refine the model while effectively working together, regardless of our physical locations.

To enhance the robustness of our model, we utilized another additional dataset prepared by the organizer. This enabled us to evaluate the model's performance across various data inputs, ensuring its ability to generalize effectively and be relevant in different sales scenario. By incorporating these datasets, we aimed to validate our methodology and deliver a flexible solution that can be leveraged by other models.

Ultimately, our goal as a team was to create a robust, reproducible model capable of accurately forecasting future sales trends. We documented the entire process, from feature selection to model evaluation, ensuring that our approach was transparent and easy to follow.

## Problem Statement
In the fast-paced retail environment, accurate sales forecasting is vital for efficient inventory management, resource allocation, and strategic decision-making. This competition encourages data scientists and machine learning enthusiasts to develop reliable and scalable models that can effectively predict retail sales trends.

Our group was tasked with developing a model to predict sales for the third year by analyzing patterns from the first two years. The model needed to consider critical factors such as seasonality, trends, and other relevant influences affecting sales.

We were encouraged to apply various feature engineering techniques, experiment with different model architectures, and implement relevant evaluation metrics to ensure the model’s predictions were as accurate as possible. Our group worked collaboratively to optimize the solution, providing a reliable and well-documented approach to future sales forecasting.

The challenge required our group to create a machine learning model capable of forecasting sales for the third year based on historical data from the first two years. This involved handling various patterns, including seasonal effects, holidays, and market shifts. We needed to develop a model that accurately captured these trends and provided reliable predictions for future sales.

Our primary goal was to develop an accurate model that could forecast third-year sales based on the available data from the first two years. We aimed to incorporate relevant factors such as seasonality and trends to ensure reliable predictions. Instead of relying on specific evaluation metrics like R-squared, we chose to evaluate the model directly based on its output and predictive capability. We assessed the model’s effectiveness by examining how well it predicted sales compared to the actual results.

## Quick Start
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. [Any other setup instructions]

## Data Overview

### Dataset 1 Summary
The dataset contains transactional records with the following structure:

- **Total Rows:** 175,514
- **Total Columns:** 6

#### Column Details
| Column Name   | Data Type | Non-Null Count | Description                                   |
|---------------|-----------|----------------|-----------------------------------------------|
| `InvoiceId`   | int64     | 175,514        | Unique identifier for each transaction.       |
| `Date`        | object    | 175,514        | The date of the transaction (initially in string format). |
| `CustomerId`  | object    | 175,514        | Unique identifier for each customer.         |
| `ProductId`   | object    | 175,130        | Unique identifier for each product.          |
| `Quantity`    | float64   | 175,130        | The number of products sold in the transaction. |
| `Amount`      | float64   | 175,130        | The total amount for the transaction.        |

#### Memory Usage
The dataset occupies approximately **8.0 MB** in memory.

#### Missing Values
The dataset contains **384 missing values** in the `ProductId`, `Quantity`, and `Amount` columns, which will require attention during preprocessing.

#### Negative Values
Upon inspection, there are **482 negative values** in the `Amount` column, which will need to be addressed in the data cleaning process.

### Dataset 2 Summary
The dataset contains transactional records with the following structure:

- **Total Rows:** 417,319
- **Total Columns:** 6

#### Column Details
| Column Name   | Data Type | Non-Null Count | Description                                   |
|---------------|-----------|----------------|-----------------------------------------------|
| `InvoiceId`   | int64     | 417,319        | Unique identifier for each transaction.       |
| `Date`        | object    | 417,319        | The date of the transaction (initially in string format). |
| `CustomerId`  | object    | 417,319        | Unique identifier for each customer.         |
| `ProductId`   | object    | 417,319        | Unique identifier for each product.          |
| `Quantity`    | int64     | 417,319        | The number of products sold in the transaction. |
| `Amount`      | float64   | 417,319        | The total amount for the transaction.        |

#### Memory Usage
The dataset occupies approximately **19.1 MB** in memory.

#### Missing Values
A preliminary check for missing values shows that all columns have non-null entries, indicating no missing data in the dataset.

#### Negative Values
Upon inspection, there is **1 negative value** in the `Amount` column, which may require further attention during preprocessing.

---

## Data Preprocessing

### Dataset 1

#### 1. Date Standardization

```python
# Remove '00:00' string from the Date column
df['Date'] = df['Date'].apply(lambda x: x.replace(' 00:00', '') if isinstance(x, str) and '00:00' in x else x)

# Convert Date to datetime format
df['DATE'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

# Sort the dataframe by date
df.sort_values(by='DATE', inplace=True)
```

**Insight:** This step ensures consistent date formatting, allowing for proper time-based analysis. The original date format contained unwanted time information, which could lead to inconsistencies. By converting the `Date` column to a standardized `datetime` format, we enhance the dataset's usability for time-series analysis.

#### 2. Handling Missing Values

```python
# Remove rows with null values in key columns
df = df.dropna(subset=['ProductId', 'Quantity', 'Amount'])
```

**Insight:** The dataset originally contained **384 rows** with missing values in critical columns (`ProductId`, `Quantity`, and `Amount`). Removing these rows ensures data integrity, allowing for more accurate analysis and predictions. These affected rows offer no benefit so it is safe to remove them.

#### 3. Column Management

```python
# Drop the original 'Date' column
df = df.drop('Date', axis=1)

# Reorder columns to make 'DATE' the first column
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
```

**Insight:** Streamlines the dataset by removing the redundant original `Date` column. The `DATE` column, which is standardized and essential for our analysis, is moved to the forefront for improved readability. Such organization enhances the dataset's clarity and makes it easier to analyze.

#### 4. Handling Negative Values

```python
# Check for negative values in Amount
negative_amount_count = (df['Amount'] < 0).sum()
print(f"Number of negative values in 'Amount' column: {negative_amount_count}")

# Remove rows with negative amount
df = df[df['Amount'] >= 0]
```

**Insight:** A total of **482 transactions** with negative amounts were identified and removed. These negative values likely indicate returns or adjustments, which are not relevant to our focus on positive sales transactions. By eliminating these records, we refine our dataset for predictive modeling, ensuring that only valid sales transactions are considered.

####  Final Dataset Summary

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Index: 174648 entries, 86401 to 145130
Data columns (total 6 columns):
 #   Column      Non-Null Count   Dtype         
---  ------      --------------   -----         
 0   DATE        174648 non-null  datetime64[ns]
 1   InvoiceId   174648 non-null  int64         
 2   CustomerId  174648 non-null  object        
 3   ProductId   174648 non-null  object        
 4   Quantity    174648 non-null  float64       
 5   Amount      174648 non-null  float64       
dtypes: datetime64 , float64(2), int64(1), object(2)
memory usage: 9.3+ MB
```

#### Key Changes
1. **Rows Reduced:** From 175,514 to 174,648 (a reduction of **866 rows**).
2. **Columns:** Reduced from 7 to 6 (removed the original `Date` column).
3. **Data Types:** 
   - DATE: `datetime64[ns]`
   - InvoiceId: `int64`
   - CustomerId: `object`
   - ProductId: `object`
   - Quantity: `float64`
   - Amount: `float64`

####  Final Memory Usage
The preprocessed dataset now occupies approximately **9.3+ MB** in memory.

####  Conclusion

The preprocessing steps have resulted in a cleaner, more consistent dataset:
- **Standardized date format** for time-series analysis enhances reliability.
- **Removed transactions with missing critical information** ensures data integrity.
- **Eliminated negative sales amounts** focuses analysis on valid sales data.
- **Optimized column order** enhances dataset readability.

This preprocessed dataset is now ready for exploratory data analysis and model development.

### Dataset 2 (start sini utk second)

#### 1. Date Standardization


## Data Analysis

```python
unique = df.nunique()
unique
```
```
Dataset 1
DATE            819
InvoiceId     57260
CustomerId      883
ProductId       231
Quantity        297
Amount         6514
```
```
Dataset 2
DATE            1042
InvoiceId     212591
CustomerId      2538
ProductId        518
Quantity        1348
Amount         21363
```

-

### Model Development
- Model Selection:
  - Rationale for chosen algorithm(s)
  - Comparison of different models (if applicable)
- Model Training:
  - Training process description
  - Hyperparameter tuning (if applicable)
- Model Evaluation:
  - Metrics used
  - Performance on validation set (third year of Dataset 1)
  - Cross-validation results (if applicable)
- Model Robustness:
  - Performance on Dataset 2
  - Comparison of results between Dataset 1 and Dataset 2
  - Discussion on model's ability to handle different conditions

### Tech Stack Used
- Programming Language(s): [List and justify choices]
- Libraries and Frameworks: [List with brief descriptions of their roles]
- Development Environment: [Describe your setup]

### Challenges and Solutions
- Challenge 1: [Name of Challenge]
  - Description
  - Solution implemented
  - Lessons learned
- Challenge 2: [Name of Challenge]
  [Repeat structure]

### Future Improvements
- Potential Enhancements: [List and briefly describe each potential improvement]
- Additional Features or Data: [Describe what could be added to improve performance]
- Scalability Considerations: [Discuss how the model could be scaled for larger datasets]

## Results
[Summary of key findings and model performance]

## Contributors
[Your name or team members]

## License
[If applicable]
