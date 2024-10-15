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

#### 2. Column Management

```python
# Drop the original 'Date' column
df = df.drop('Date', axis=1)

# Reorder columns to make 'DATE' the first column
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
```

**Insight:** Streamlines the dataset by removing the redundant original `Date` column. The `DATE` column, which is standardized and essential for our analysis, is moved to the forefront for improved readability. Such organization enhances the dataset's clarity and makes it easier to analyze.

### Dataset 1

#### 3. Handling Missing Values

```python
# Remove rows with null values in key columns
df = df.dropna(subset=['ProductId', 'Quantity', 'Amount'])
```
**Insight:** The dataset originally contained **384 rows** with missing values in critical columns (`ProductId`, `Quantity`, and `Amount`). Removing these rows ensures data integrity, allowing for more accurate analysis and predictions. These affected rows offer no benefit so it is safe to remove them.

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

### Dataset 2 

### Dataset 2 

#### 1. Handling Negative Values

```python
# Check for negative values in Amount
negative_amount_count = (df['Amount'] < 0).sum()
print(f"Number of negative values in 'Amount' column: {negative_amount_count}")

# Remove rows with negative amount
df = df[df['Amount'] >= 0]
```

**Insight:** We identified and removed **1 transaction** with a negative amount. This negative value probably indicates a return or adjustment, which is irrelevant to our emphasis on positive sales transactions. By discarding this record, we improve our dataset for predictive modeling, ensuring that it only includes valid sales transactions.

####  Final Dataset Summary

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Index: 417318 entries, 417318 to 1050
Data columns (total 6 columns):
 #   Column      Non-Null Count   Dtype         
---  ------      --------------   -----         
 0   InvoiceId   417318 non-null  int64         
 1   CustomerId  417318 non-null  object        
 2   ProductId   417318 non-null  object        
 3   Quantity    417318 non-null  int64         
 4   Amount      417318 non-null  float64       
 5   DATE        417318 non-null  datetime64[ns]
dtypes: datetime64[ns](1), float64(1), int64(2), object(2)
memory usage: 22.3+ MB
```

#### Key Changes
1. **Rows Reduced:** From 417,319 to 417,318 (a reduction of **1 rows**).
2. **Columns:** Reduced from 7 to 6 (removed the original `Date` column).
3. **Data Types:** 
   - DATE: `datetime64[ns]`
   - InvoiceId: `int64`
   - CustomerId: `object`
   - ProductId: `object`
   - Quantity: `float64`
   - Amount: `float64`

####  Final Memory Usage
The preprocessed dataset now occupies approximately **22.3+ MB** in memory.

####  Conclusion

The preprocessing steps have resulted in a cleaner, more consistent dataset:
- **Standardized date format** for time-series analysis enhances reliability.
- **Removed transactions with missing critical information** ensures data integrity.
- **Eliminated negative sales amounts** focuses analysis on valid sales data.
- **Optimized column order** enhances dataset readability.

This preprocessed dataset is now ready for exploratory data analysis and model development.

## Data Analysis

```python
unique = df.nunique()
unique
```
![image](https://github.com/user-attachments/assets/0d69f688-39c2-41a4-b350-58c593411460)
- Dataset 1

![image](https://github.com/user-attachments/assets/06dd6ce2-c27e-4aca-8138-a591b0691389)
- Dataset 2

#### Insights:
- Data Volume: Dataset 2 is significantly larger, with 417,318 records compared to 174,648 in Dataset 1.
- Customer Base: Dataset 2 has a larger customer base (2,538 unique customers) compared to Dataset 1 (883 unique customers).
- Product Variety: Dataset 2 offers more products (518 unique ProductIds) than Dataset 1 (231 unique ProductIds).
- Date Range: Dataset 2 covers a longer period (1,042 unique dates) compared to Dataset 1 (819 unique dates).

### Monthly Sales Analysis (2021-2023)
- Dataset 1
```
=== Monthly Sales Analysis (2021-2023) ===
--------------------------------------------------
Year  Month           Total Sales    % Change
--------------------------------------------------
--------------------------------------------------
2021  January     $  6,196,261.70            
2021  February    $  6,428,423.37       +3.7%
2021  March       $  4,701,558.70      -26.9%
2021  April       $  4,774,511.37       +1.6%
2021  May         $  2,732,043.96      -42.8%
2021  June        $  1,959,853.77      -28.3%
2021  July        $  1,687,258.09      -13.9%
2021  August      $  3,231,887.77      +91.5%
2021  September   $  3,211,008.03       -0.6%
2021  October     $  5,107,857.98      +59.1%
2021  November    $  4,710,511.72       -7.8%
2021  December    $  6,815,252.45      +44.7%
--------------------------------------------------
2022  January     $ 13,900,962.83     +104.0%
2022  February    $  6,402,649.04      -53.9%
2022  March       $  4,257,504.66      -33.5%
2022  April       $  4,866,249.15      +14.3%
2022  May         $  3,983,449.40      -18.1%
2022  June        $  5,990,105.69      +50.4%
2022  July        $  8,258,821.64      +37.9%
2022  August      $  3,912,611.64      -52.6%
2022  September   $  5,069,485.43      +29.6%
2022  October     $  5,238,628.12       +3.3%
2022  November    $  7,078,806.50      +35.1%
2022  December    $  7,126,982.18       +0.7%
--------------------------------------------------
2023  January     $ 14,463,245.37     +102.9%
2023  February    $  3,553,592.61      -75.4%
2023  March       $  5,195,484.25      +46.2%
2023  April       $  3,383,719.12      -34.9%
2023  May         $  5,810,563.66      +71.7%
2023  June        $  5,354,211.65       -7.9%
2023  July        $  3,933,952.55      -26.5%
2023  August      $  6,987,626.75      +77.6%
2023  September   $  5,898,829.16      -15.6%
2023  October     $  5,348,591.11       -9.3%
2023  November    $  6,337,921.89      +18.5%
2023  December    $  8,365,375.34      +32.0%

Summary Statistics:
--------------------------------------------------
Total Sales: $202,275,798.65
Average Monthly Sales: $5,618,772.18
Highest Month: January 2023 ($14,463,245.37)
Lowest Month: July 2021 ($1,687,258.09)
```
- Dataset 2
```
=== Monthly Sales Analysis (2021-2023) ===
--------------------------------------------------
Year  Month           Total Sales    % Change
--------------------------------------------------
--------------------------------------------------
2021  January     $  4,268,740.98            
2021  February    $  3,304,584.09      -22.6%
2021  March       $  4,539,463.82      +37.4%
2021  April       $  3,754,174.73      -17.3%
2021  May         $  2,965,499.52      -21.0%
2021  June        $  3,097,231.48       +4.4%
2021  July        $  3,390,558.57       +9.5%
2021  August      $  3,105,933.13       -8.4%
2021  September   $  4,018,885.17      +29.4%
2021  October     $  4,634,259.17      +15.3%
2021  November    $  4,597,474.04       -0.8%
2021  December    $  4,388,728.45       -4.5%
--------------------------------------------------
2022  January     $  2,918,894.61      -33.5%
2022  February    $  3,503,575.99      +20.0%
2022  March       $  3,464,302.74       -1.1%
2022  April       $  3,907,407.72      +12.8%
2022  May         $  4,015,319.15       +2.8%
2022  June        $  5,578,820.95      +38.9%
2022  July        $  3,245,442.59      -41.8%
2022  August      $  3,471,612.21       +7.0%
2022  September   $  3,614,950.34       +4.1%
2022  October     $  3,658,879.75       +1.2%
2022  November    $  4,520,220.18      +23.5%
2022  December    $  4,366,027.69       -3.4%
--------------------------------------------------
2023  January     $  3,272,823.60      -25.0%
2023  February    $  4,411,867.04      +34.8%
2023  March       $  4,991,619.02      +13.1%
2023  April       $  3,449,348.64      -30.9%
2023  May         $  4,431,958.73      +28.5%
2023  June        $  4,524,782.30       +2.1%
2023  July        $  5,470,284.56      +20.9%
2023  August      $  4,880,602.40      -10.8%
2023  September   $  4,656,024.36       -4.6%
2023  October     $  5,285,113.99      +13.5%
2023  November    $  5,050,258.57       -4.4%
2023  December    $  5,549,126.04       +9.9%

Summary Statistics:
--------------------------------------------------
Total Sales: $148,304,796.32
Average Monthly Sales: $4,119,577.68
Highest Month: June 2022 ($5,578,820.95)
Lowest Month: January 2022 ($2,918,894.61)
```
### Product Co-Purchase Analysis
- Dataset 1
```
Top 10 Most Frequently Co-Purchased Product Pairs:
------------------------------------------------------------
Rank  Product Pair                       Purchase Count % of Total
------------------------------------------------------------
1     P-279, P-301                       443                 54.09%
2     P-196, P-301                       439                 53.60%
3     P-196, P-279                       439                 53.60%
4     P-271, P-278                       437                 53.36%
5     P-301, P-62587                     437                 53.36%
6     P-279, P-62587                     437                 53.36%
7     P-271, P-62587                     436                 53.24%
8     P-278, P-62587                     435                 53.11%
9     P-167, P-278                       435                 53.11%
10    P-167, P-271                       435                 53.11%

Summary Statistics:
------------------------------------------------------------
Total number of transactions analyzed: 819
Total unique product pairs found: 19202
Average co-purchase frequency: 34.31
Median co-purchase frequency: 8.00
Maximum co-purchase frequency: 443
Minimum co-purchase frequency: 1
```
- Dataset 2
```
=== Product Co-Purchase Analysis ===

Top 10 Most Frequently Co-Purchased Product Pairs:
------------------------------------------------------------
Rank  Product Pair                       Purchase Count % of Total
------------------------------------------------------------
1     P-795484, P-796828                 538                 51.63%
2     P-795484, P-795965                 519                 49.81%
3     P-1883364, P-795484                519                 49.81%
4     P-795298, P-795484                 504                 48.37%
5     P-795484, P-795558                 503                 48.27%
6     P-1569690, P-795484                502                 48.18%
7     P-795484, P-795710                 499                 47.89%
8     P-795484, P-795581                 494                 47.41%
9     P-795350, P-795484                 490                 47.02%
10    P-795965, P-796828                 486                 46.64%

Summary Statistics:
------------------------------------------------------------
Total number of transactions analyzed: 1042
Total unique product pairs found: 95307
Average co-purchase frequency: 30.04
Median co-purchase frequency: 8.00
Maximum co-purchase frequency: 538
Minimum co-purchase frequency: 1
```
### Average Time Between Purchases (Shortest to Longest)
- Dataset 1
```
Top 10 Average Time Between Purchases (Shortest to Longest):
------------------------------------------------
Customer ID | Average Days Between Purchases
------------------------------------------------
 D-108768   | 0.000 days
  D-17314   | 0.000 days
  D-17384   | 0.000 days
  D-17849   | 0.000 days
  D-17875   | 0.000 days
 D-226898   | 0.000 days
 D-312006   | 0.000 days
 D-347004   | 0.000 days
  D-35880   | 0.000 days
 D-370412   | 0.000 days

Summary Statistics:
Shortest interval: 0.000 days
Longest interval: 635.000 days
Overall average: 17.346 days
Total customers analyzed: 851
```
- Dataset 2
```
Average Time Between Purchases (Shortest to Longest):
------------------------------------------------
Customer ID | Average Days Between Purchases
------------------------------------------------
 D-1014566  | 0.000 days
 D-1042001  | 0.000 days
 D-1046734  | 0.000 days
 D-1047026  | 0.000 days
 D-1047027  | 0.000 days
 D-1053067  | 0.000 days
 D-1053087  | 0.000 days
 D-1055535  | 0.000 days
 D-1061004  | 0.000 days
 D-1062306  | 0.000 days

Summary Statistics:
Shortest interval: 0.000 days
Longest interval: 940.000 days
Overall average: 36.069 days
Total customers analyzed: 2411
```
### Customer Segmentation Analysis
### Top 10 Products by Total Quantity Sold and Total Revenue Generated


### Plot Total Amount vs Month to see the overall trend from 2021 to 2023
- Dataset 1

![image](https://github.com/user-attachments/assets/135d0fbc-536f-45a6-9c32-8152829521e1)

The overall trend shows growth from 2021 to 2023. Starting around 6 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in early 2022 reaching around 14 million, and late 2023 showing similar highs. Despite some dips, each year tends to end at a higher point than it started. This indicates a positive growth trajectory for the business over the observed period.

- Dataset 2

![image](https://github.com/user-attachments/assets/a0e55358-f87d-4021-a314-268d60783897)

The overall trend shows growth from 2021 to 2023. Starting around 4 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in mid-2022 reaching 5.5 million, and late 2023 showing similar highs. A the end of 2023 maintains these higher levels around 5-5.5 million. Despite some dips, each year tends to end at a higher point than it started.

### Analyze what day of the week has the highest sales

### Total sales by month for all 2021-2023 transactions

### Total Revenue by Month and Day of the Week

### Total Quantity trend (by month) from 2021 to 2023

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
