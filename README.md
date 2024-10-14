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

## Data Preprocessing Steps

### 1. Date Standardization

```python
# Remove '00:00' string from the Date column
df['Date'] = df['Date'].apply(lambda x: x.replace(' 00:00', '') if isinstance(x, str) and '00:00' in x else x)

# Convert Date to datetime format
df['DATE'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

# Sort the dataframe by date
df.sort_values(by='DATE', inplace=True)
```
- Data Cleaning: [Describe techniques used]
- Feature Engineering: [Explain process and rationale]
- Handling Missing Data: [If applicable, describe approach]
- Data Normalization/Scaling: [Describe methods used and why]

### Data Analysis
-
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
