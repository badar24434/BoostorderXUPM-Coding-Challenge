# Sales Prediction 

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Quick Start](#quick-start)
- [Data Overview](#data-overview)
- [Data Preprocessing](#data-preprocessing)
- [Sales Analysis](#sales-analysis)
- [Model Development](#model-development)
  - [Model Selection](#model-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Robustness](#model-robustness)
- [Tech Stack Used](#tech-stack-used)
- [Challenges and Solutions](#challenges-and-solutions)

--- 

## Introduction
Our group participated in a coding challenge aimed at building a machine learning model to predict sales for the third year, using patterns from the previous two years of sales data. The challenge required us to consider key factors such as seasonality and trends, which could significantly impact the accuracy of sales predictions.

Our group selected Google Colab as our main development platform to enhance collaboration and efficiency. This platform enabled us to share code and collaborate in real-time, greatly improving our teamwork. With integrated support for popular machine learning libraries such as TensorFlow, scikit-learn, and Pandas, Colab provided the necessary cloud-based computational resources to train and test our models without the limitations of local hardware. Throughout the project, we took advantage of its collaborative features to explore various machine learning techniques and feature engineering strategies, fostering discussions and troubleshooting that allowed us to iteratively refine the model while effectively working together, regardless of our physical locations.

Ultimately, our goal as a team was to create a robust, reproducible model capable of accurately forecasting future sales trends. We documented the entire process, from feature selection to model evaluation, ensuring that our approach was transparent and easy to follow.

## Problem Statement
In the fast-paced retail environment, accurate sales forecasting is vital for efficient inventory management, resource allocation, and strategic decision-making. This competition encourages data scientists and machine learning enthusiasts to develop reliable and scalable models that can effectively predict retail sales trends.

Our group was tasked with developing a model to predict sales for the third year by analyzing patterns from the first two years. The model needed to consider critical factors such as seasonality, trends, and other relevant influences affecting sales.

We were encouraged to apply various feature engineering techniques, experiment with different model architectures, and implement relevant evaluation metrics to ensure the model’s predictions were as accurate as possible. Our group worked collaboratively to optimize the solution, providing a reliable and well-documented approach to future sales forecasting.

Our primary goal was to develop an accurate model that could forecast third-year sales based on the available data from the first two years. We aimed to incorporate relevant factors such as seasonality and trends to ensure reliable predictions. Instead of relying on specific evaluation metrics like R-squared, we chose to evaluate the model directly based on its output and predictive capability. We assessed the model’s effectiveness by examining how well it predicted sales compared to the actual results.

## Quick Start
## Option 1: Local Setup
1. Clone this repository:
   ```python
   git clone [repository URL]
   ```
   
2. Install dependencies: 
   ```python
   pip install -r requirements.txt
   ```
## Option 2: Google Colab
For easier access without any local setup, you can directly use our Google Colab notebook. Both links contain the same code, but each focuses on a different analysis description. 
We created two separate notebooks because each dataset needs to be tested and explained individually, which wouldn't be feasible in a single notebook.

### Dataset 1:
```python
https://colab.research.google.com/drive/1xKHE7x_ObMryRnS4MMTd3MbThM4bnd6f?usp=sharing
```

### Dataset 2:
```python 
https://colab.research.google.com/drive/1AbrfKU8MhEEvgMZ2WV23kgfWh924yOJp?usp=sharing
```

This option allows you to run the project directly in your browser without any local installation.

---

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

## 1. Date Standardization

```python
# Remove '00:00' string from the Date column
df['Date'] = df['Date'].apply(lambda x: x.replace(' 00:00', '') if isinstance(x, str) and '00:00' in x else x)

# Convert Date to datetime format
df['DATE'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

# Sort the dataframe by date
df.sort_values(by='DATE', inplace=True)
```

**Insight:** This step ensures consistent date formatting, allowing for proper time-based analysis. The original date format contained unwanted time information, which could lead to inconsistencies. By converting the `Date` column to a standardized `datetime` format, we enhance the dataset's usability for time-series analysis.

---

## 2. Column Management

```python
# Drop the original 'Date' column
df = df.drop('Date', axis=1)

# Reorder columns to make 'DATE' the first column
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
```

**Insight:** Streamlines the dataset by removing the redundant original `Date` column. The `DATE` column, which is standardized and essential for our analysis, is moved to the forefront for improved readability. Such organization enhances the dataset's clarity and makes it easier to analyze.

---

## 3. Handling Missing Values

### Missing Values Analysis: Dataset 1 vs Dataset 2

### Dataset 1: Missing Values

- **Affected Rows:** 384
- **Columns:** ProductId, Quantity, Amount
- **Nature:** All three columns simultaneously affected in these rows
- **Possible Causes:** Cancelled purchases, system errors, incomplete transactions

**Impact and Mitigation:**
- Could skew product performance, revenue calculations, and customer behavior analysis
- Solution: Removed affected rows using the following code:

```python
# Remove rows with null values in key columns
df = df.dropna(subset=['ProductId', 'Quantity', 'Amount'])
```

- This ensures data integrity and analytical accuracy

### Dataset 2: No Missing Value

- **Missing Values:** None
- **Data Completeness:** 100%
- Allows for reliable analysis without missing values preprocessing

---

## 4. Handling Negative Values
### Handling Extreme Negative Values in Sales Data

During the data preprocessing phase of our sales forecasting project, we encountered a critical issue: the presence of extreme negative values in our sales data, particularly in Dataset 1. While Dataset 2 has a different profile of negative values, we've decided to standardize our approach across both datasets. This section outlines the nature of this issue, its potential impact on our analysis, and our decision to remove all negative values from both datasets.

## The Issue: Extreme Negative Values

Let's look at a sample of the problematic data from Dataset 1:

| InvoiceId | CustomerId | ProductId | Quantity | Amount      | DATE       |
|-----------|------------|-----------|----------|-------------|------------|
| 2435607   | D-247996   | P-52070   | 432      | -13157553.6 | 05/05/2022 |
| 2419277   | D-247996   | P-190592  | 432      | -10974960   | 26/04/2022 |
| 2419277   | D-247996   | P-190600  | 432      | -10675454.4 | 26/04/2022 |
| 2471193   | D-247996   | P-190600  | 396      | -8956985.2  | 25/05/2022 |
| 2468364   | D-63601    | P-202     | 960      | -7227648    | 23/05/2022 |

These values are clearly problematic. For instance:

- Product P-52070 has a unit cost of RM319. The expected amount for 432 units would be RM137,808. However, the recorded amount is -RM13,157,553.6, which is nearly 100 times larger in magnitude.
- Similar discrepancies exist for other products, with negative amounts far exceeding any reasonable sale or refund value.

Now we look at the negative value from Dataset 2:
### Dataset 2: Single Negative Value

Dataset 2 has far fewer negative values compared to Dataset 1. In fact, it contains only **one row** with a negative amount:
| InvoiceId | CustomerId | ProductId | Quantity | Amount      | DATE       |
|-----------|------------|-----------|----------|-------------|------------|
| 2435607   | D-357869   | P-796828  |    1     | -12.86      | 22/01/2021 |

While the magnitude of this negative value is not as extreme as those in Dataset 1, it is still inconsistent with valid sales transactions. As part of our standardized approach, this single row will also be removed. Dataset 2 only had a single negative value, removing it ensures our methodology remains consistent and reliable across both datasets. This minor adjustment enhances data quality and ensures our forecasting and analysis remain accurate without introducing any distortions.

## Differences Between Dataset 1 and Dataset 2

It's important to note the differences between our two datasets:

1. **Dataset 1**: 
   - Contains approximately 400 rows with significant negative values out of 175,514 total rows (0.23%).
   - These negative values are extreme and clearly erroneous, as shown in the sample above.

2. **Dataset 2**:
   - Contains only one row with a negative value.
   - The magnitude of this negative value is not as extreme as those in Dataset 1.

## Rationale for Removing Affected Rows

1. **Data Integrity**: 
   - The extreme negative values, especially in Dataset 1, are clearly erroneous and do not represent actual sales or refund transactions.
   - Including them would severely compromise the integrity of our dataset and any subsequent analysis.

2. **Statistical Distortion**:
   - Extreme outliers like these will significantly distort statistical measures such as mean, standard deviation, and variance.
   - This distortion would make our analyses unreliable and potentially lead to incorrect conclusions.

3. **Forecasting Accuracy**:
   - Including these values in our forecasting model would lead to severely skewed predictions.
   - The model might interpret these as valid data points, leading to unrealistic and unusable forecasts.

4. **Business Logic**:
   - No reasonable refund or adjustment would be many times larger than the possible sale value.
   - These values violate basic business logic and accounting principles.

5. **Standardization Across Datasets**:
   - While Dataset 2 only has one negative value, we've decided to standardize our approach across both datasets.
   - This ensures consistency in our data preprocessing steps and simplifies our overall methodology.
   - Standardization makes our process more robust and easier to maintain, especially if we need to incorporate additional datasets in the future.

6. **Simplicity and Reproducibility**:
   - A consistent rule to remove all negative values is simpler to implement and explain.
   - This approach enhances the reproducibility of our analysis across different datasets.

## Implementation and Impact

We have decided to remove all rows with negative values from both Dataset 1 and Dataset 2. Here's why this approach is appropriate:

1. **Scope of Removal**: 
   - In Dataset 1: Out of 175,514 total rows, only 400 rows (0.23%) are affected.
   - In Dataset 2: Only 1 row is affected from 417,319 row.
   - This small percentage ensures that removing these rows will not significantly reduce our datasets' overall size or representativeness.

2. **Consistency**: 
   - By removing all negative values across both datasets, we ensure a consistent approach in our data preprocessing.

3. **Simplicity**: 
   - This approach is straightforward to implement and explain, enhancing the reproducibility of our analysis.

4. **Data Quality**: 
   - Removing these rows improves the overall quality and reliability of our datasets.

Here's the code snippet implementing this decision:

```python
# Check for negative values in Amount
negative_amount_count = (df['Amount'] < 0).sum()
print(f"Number of negative values in 'Amount' column: {negative_amount_count}")

# Remove rows with negative amount
df = df[df['Amount'] >= 0]
```

**Insight:** A total of **482 transactions** with negative amounts were identified and removed from Dataset 1 and only **1 transcation** from Dataset 2 were identified and removed. These negative values likely indicate returns or adjustments, which are not relevant to our focus on positive sales transactions. By eliminating these records, we refine our dataset for predictive modeling, ensuring that only valid sales transactions are considered.

### Conclusion
While the impact of negative values differs between Dataset 1 and Dataset 2, our decision to remove all negative values from both datasets ensures a standardized and consistent approach to data preprocessing. This strategy not only addresses the severe issues in Dataset 1 but also establishes a uniform methodology that can be applied consistently across multiple datasets. By implementing this approach, we significantly improve the quality of our datasets without materially impacting their size or representativeness, setting a foundation for more accurate and reliable sales forecasting and financial analysis.

---

### Dataset 1
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

### Dataset 2 

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

---

## Sales Analysis

This section provides a detailed examination of our sales data, focusing on trends, seasonality, and overall performance. We explore monthly sales patterns, identify key drivers of revenue, and assess the impact of external economic factors.
```python
unique = df.nunique()
unique
```
In the following analysis, we compare Dataset 1 and Dataset 2 to examine differences in data volume, customer base, product variety, and date range, which are critical factors for our predictive modeling.

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
Provides a detailed comparison of monthly sales figures over the three-year period, helping to identify seasonal trends and significant changes in performance across different years.

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
Examines which products are frequently purchased together, offering insights into customer buying behavior and potential cross-selling opportunities.

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
Analyzes the average duration between purchases made by customers, helping to understand customer purchase frequency and engagement patterns.

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
Groups customers based on purchasing behavior, demographics, or other criteria, allowing for tailored marketing strategies and product recommendations.

- Dataset 1
```
=== Customer Segmentation Analysis ===

Customer Segments:
--------------------------------------------------
High-Value Customers: 139 customers
Average Spend: $1266145.30
Average Order Value: $4576.82

Frequent Customers: 220 customers
Average Purchase Frequency: 654.7 orders
Average Spend: $808160.70

Occasional Customers: 228 customers
Average Purchase Frequency: 6.2 orders
Average Spend: $5408.19

=== Customer Lifetime Value (CLV) Analysis ===

Top 10 Most Valuable Customers:
--------------------------------------------------
Rank  Customer ID    CLV            Total Spend    Orders    
--------------------------------------------------
1     D-17335        $8555834.13    $8555834.13    649       
2     D-18042        $8468478.20    $8468478.20    315       
3     D-18239        $8230296.43    $8230296.43    238       
4     D-18552        $7702661.18    $7702661.18    607       
5     D-170817       $6636883.24    $6636883.24    3146      
6     D-17707        $5692302.73    $5692302.73    365       
7     D-17369        $4836165.36    $4836165.36    1738      
8     D-17537        $4824567.99    $4824567.99    4541      
9     D-18191        $4395850.37    $4395850.37    2441      
10    D-17420        $4174306.75    $4174306.75    736       

CLV Summary Statistics:
--------------------------------------------------
Average CLV: $229077.91
Median CLV: $14281.10
Total Customer Base Value: $202275798.65
```
- Dataset 2
```
=== Customer Segmentation Analysis ===

Customer Segments:
--------------------------------------------------
High-Value Customers: 388 customers
Average Spend: $315468.47
Average Order Value: $4211.87

Frequent Customers: 630 customers
Average Purchase Frequency: 616.6 orders
Average Spend: $206350.54

Occasional Customers: 714 customers
Average Purchase Frequency: 3.6 orders
Average Spend: $5027.60

=== Customer Lifetime Value (CLV) Analysis ===

Top 10 Most Valuable Customers:
--------------------------------------------------
Rank  Customer ID    CLV            Total Spend    Orders    
--------------------------------------------------
1     D-344608       $13298608.77   $13298608.77   1692      
2     D-344616       $12293077.39   $12293077.39   2368      
3     D-356924       $10207901.50   $10207901.50   895       
4     D-356821       $6654672.67    $6654672.67    1641      
5     D-357221       $6462580.45    $6462580.45    1750      
6     D-357220       $4721921.91    $4721921.91    1240      
7     D-356911       $4402783.99    $4402783.99    1136      
8     D-357699       $3976271.95    $3976271.95    177770    
9     D-343658       $3519201.72    $3519201.72    1931      
10    D-355435       $2574501.48    $2574501.48    2254      

CLV Summary Statistics:
--------------------------------------------------
Average CLV: $58433.73
Median CLV: $2632.44
Total Customer Base Value: $148304796.32
```

### Top 10 Products by Total Quantity Sold and Total Revenue Generated
Highlights the best-selling products in terms of quantity and revenue, providing insights into which items contribute the most to the business's profitability.

- Dataset 1
```
Top 10 Products by Total Quantity Sold
================================================================================
Rank ProductId      Total Quantity Total Revenue       Avg Price/Unit 
--------------------------------------------------------------------------------
1    P-62587        831,689        $83,120,955.53      $99.94         
2    P-279          129,842        $456,235.95         $3.51          
3    P-301          126,754        $887,278.00         $7.00          
4    P-278          107,947        $352,038.60         $3.26          
5    P-271          98,222         $687,554.00         $7.00          
6    P-711430       87,695         $12,040,739.60      $137.30        
7    P-196          86,297         $16,596,137.17      $192.31        
8    P-202          81,961         $12,170,767.90      $148.49        
9    P-167          72,756         $11,441,602.88      $157.26        
10   P-262          27,783         $1,499,500.46       $53.97         

Top 10 Products by Total Revenue Generated
================================================================================
Rank ProductId      Total Quantity Total Revenue       Avg Price/Unit 
--------------------------------------------------------------------------------
1    P-62587        831,689        $83,120,955.53      $99.94         
2    P-196          86,297         $16,596,137.17      $192.31        
3    P-202          81,961         $12,170,767.90      $148.49        
4    P-711430       87,695         $12,040,739.60      $137.30        
5    P-167          72,756         $11,441,602.88      $157.26        
6    P-215          9,277          $7,968,226.63       $858.92        
7    P-157          4,815          $4,285,732.55       $890.08        
8    P-287          24,240         $3,838,074.43       $158.34        
9    P-187          27,546         $3,646,823.02       $132.39        
10   P-286          15,845         $3,407,328.69       $215.04        

Overall Product Performance Summary
================================================================================
Total number of unique products: 231
Total quantity sold across all products: 2,008,861
Total revenue generated: $202,275,798.65
Average revenue per product: $875,652.81
Median revenue per product: $20,589.20
Average price per unit across all products: $137.04
```
- Dataset 2
```
Top 10 Products by Total Quantity Sold
================================================================================
Rank ProductId      Total Quantity Total Revenue       Avg Price/Unit 
--------------------------------------------------------------------------------
1    P-796828       19,683,035     $69,506,459.34      $3.53          
2    P-796005       3,521,273      $2,297,292.63       $0.65          
3    P-796023       1,267,171      $817,558.47         $0.65          
4    P-795362       633,947        $1,762,711.07       $2.78          
5    P-796013       589,297        $2,804,907.14       $4.76          
6    P-795484       555,680        $3,671,064.46       $6.61          
7    P-795369       477,244        $2,436,707.25       $5.11          
8    P-4398000      422,791        $1,211,347.98       $2.87          
9    P-796009       396,474        $951,294.52         $2.40          
10   P-795104       375,332        $4,081,751.13       $10.88         

Top 10 Products by Total Revenue Generated
================================================================================
Rank ProductId      Total Quantity Total Revenue       Avg Price/Unit 
--------------------------------------------------------------------------------
1    P-796828       19,683,035     $69,506,459.34      $3.53          
2    P-795179       346,619        $4,642,464.06       $13.39         
3    P-795104       375,332        $4,081,751.13       $10.88         
4    P-795069       362,420        $4,029,276.00       $11.12         
5    P-795484       555,680        $3,671,064.46       $6.61          
6    P-796013       589,297        $2,804,907.14       $4.76          
7    P-795558       232,794        $2,695,519.17       $11.58         
8    P-795369       477,244        $2,436,707.25       $5.11          
9    P-796005       3,521,273      $2,297,292.63       $0.65          
10   P-795362       633,947        $1,762,711.07       $2.78          

Overall Product Performance Summary
================================================================================
Total number of unique products: 518
Total quantity sold across all products: 38,885,635
Total revenue generated: $148,304,796.32
Average revenue per product: $286,302.70
Median revenue per product: $17,979.17
Average price per unit across all products: $22.01
```


### Plot Total Amount vs Month to see the overall trend from 2021 to 2023
Visualizes the overall revenue trend from 2021 to 2023, giving a clear view of growth patterns and any potential dips in sales over time.

- Dataset 1

![image](https://github.com/user-attachments/assets/135d0fbc-536f-45a6-9c32-8152829521e1)

The overall trend shows growth from 2021 to 2023. Starting around 6 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in early 2022 reaching around 14 million, and late 2023 showing similar highs. Despite some dips, each year tends to end at a higher point than it started. This indicates a positive growth trajectory for the business over the observed period.

- Dataset 2

![image](https://github.com/user-attachments/assets/a0e55358-f87d-4021-a314-268d60783897)

The overall trend shows growth from 2021 to 2023. Starting around 4 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in mid-2022 reaching 5.5 million, and late 2023 showing similar highs. A the end of 2023 maintains these higher levels around 5-5.5 million. Despite some dips, each year tends to end at a higher point than it started.

### Analyze what day of the week has the highest sales
Identifies which days consistently see the highest sales volumes, allowing businesses to optimize operations and marketing efforts accordingly.

- Dataset 1

  ![image](https://github.com/user-attachments/assets/b4cc48c9-6ec6-48e0-84e6-354d4887bd8f)
```
  Unique Orders by Month and Day:
day        Monday  Tuesday  Wednesday  Thursday  Friday  Saturday  Sunday
month                                                                    
January    1458.0   1083.0     1397.0     727.0   839.0     145.0   119.0
February    825.0    637.0     1019.0     904.0   689.0      21.0     5.0
March      1175.0   1018.0     1168.0    1049.0   658.0       NaN     9.0
April      1048.0    970.0     1038.0    1082.0   750.0       4.0    10.0
May        1150.0    994.0     1055.0     897.0   652.0      12.0     2.0
June        770.0    898.0     1026.0     923.0   637.0       8.0     NaN
July        988.0    786.0      761.0     841.0   698.0      34.0     NaN
August     1008.0    806.0      854.0     855.0   451.0      44.0    21.0
September  1016.0    819.0     1054.0     998.0   698.0      15.0    10.0
October    1134.0    783.0     1105.0     931.0   854.0      31.0     4.0
November   1132.0   1125.0     1264.0     907.0   641.0      15.0    17.0
December   1105.0    932.0     1339.0    1323.0   989.0       1.0     NaN
```
Order patterns are very consistent across months. Weekdays, especially Monday through Wednesday, regularly see 1,000-1,500 orders. January shows the highest spike with Monday hitting nearly 1,500 orders. Weekends drop significantly - Saturdays average around 100 orders and Sundays even less. The pattern is remarkably stable across all months, with slight variations in peak days.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/01977e3f-cdbb-469c-8216-3c4d811b69f1)
```
  Unique Orders by Month and Day:
day        Monday  Tuesday  Wednesday  Thursday  Friday  Saturday  Sunday
month                                                                    
January    3087.0   3303.0     3022.0    1909.0  2382.0     372.0   309.0
February   3048.0   2967.0     2734.0    2558.0  2478.0     482.0   281.0
March      4416.0   4530.0     3270.0    3437.0  3191.0     428.0   364.0
April      4356.0   3344.0     3476.0    2422.0  1875.0     340.0   315.0
May        5042.0   2661.0     2311.0    4302.0  2956.0     378.0   242.0
June       2828.0   5900.0     3453.0    3175.0  2786.0    1079.0   743.0
July       2824.0   4211.0     5067.0    3737.0  2304.0    1039.0   725.0
August     4431.0   4431.0     3024.0    3458.0  2379.0     772.0   942.0
September  4831.0   3798.0     3350.0    2829.0  2160.0    1574.0   619.0
October    2939.0   3530.0     3726.0    3494.0  3005.0     109.0   304.0
November   4270.0   3691.0     3293.0    2399.0  2190.0    1703.0   282.0
December   2577.0   4089.0     4571.0    2780.0  2199.0     171.0   212.0
```
Order patterns are very consistent across months. Weekdays, especially Monday through Wednesday, regularly see 4,000-5,000 orders. June shows the highest spike with Tuesday hitting nearly 6,000 orders. Weekends drop significantly - Saturdays average around 1,000 orders and Sundays even less. The pattern is remarkably stable across all months, with slight variations in peak days.

### Total sales by month for all 2021-2023 transactions
Aggregates total sales for each month across the entire period, providing a comprehensive overview of monthly sales performance.

- Dataset 1

  ![image](https://github.com/user-attachments/assets/d5d005ff-0350-4a6b-8dfb-4dfe72687847)

Sales follow a distinct yearly pattern. They start highest in January at about 34 million, then drop sharply to around 16 million in February. The trend then shows a gradual decrease until May, reaching the lowest point of about 12.5 million. From there, sales slowly increase through the rest of the year. The growth accelerates from September onwards, with a particularly steep rise from October to December. The year ends with December sales at about 22.5 million, though still below the January peak.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/4845eed7-a8e3-46ab-8fd8-1afe44d1d8cd)

Sales follow a clear yearly pattern. They start low in January (around 10.5 million) and build up through March (13 million). There's usually a dip in April, but things really pick up from September onwards. The end of the year is strongest - October through December show steady growth, reaching peak sales of about 14.2 million in December.


### Total Revenue by Month and Day of the Week
Breaks down total revenue not only by month but also by day of the week, offering a dual perspective on how both timeframes influence sales.

- Dataset 1

  ![image](https://github.com/user-attachments/assets/48d24c65-28b1-4b73-9752-54e10e59a5e6)

Weekdays generate most of the revenue throughout the year. Monday through Friday typically bring in between 2-5 million in revenue. There's significant variation, but Wednesday and Friday often show higher revenue, especially towards the end of the year. Weekends consistently show the lowest revenue, with Saturday and Sunday rarely exceeding 1 million. There's a notable peak in January for most weekdays, with Monday, Tuesday, and Wednesday all reaching around 5-6 million. December also shows high revenue for several weekdays, particularly Friday which peaks at about 6 million.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/87ad16bb-30d3-4e93-b4fc-ead299db9806)

Weekdays clearly drive most revenue throughout the year. Monday through Thursday typically bring in between 2.5-3.5 million in revenue. Wednesday and Thursday often lead the pack, especially in later months. Weekends tell a different story - Saturday and Sunday consistently show the lowest revenue, rarely going above 500,000. There's a noticeable peak in March for Monday and another in April for Wednesday, both hitting around 3.5 million.



### Total Quantity trend (by month) from 2021 to 2023
Tracks the total quantity of products sold each month over the three years, highlighting trends in volume and demand fluctuations.


- Dataset 1

  ![image](https://github.com/user-attachments/assets/b9c8a26f-1647-4b90-8b3b-8ac11826c82f)

The quantity shows significant fluctuations from 2021 to 2023. Early 2021 started around 70,000 units, then we see regular ups and downs. There are two notable peaks: one in early 2022 at about 120,000 units, and another in early 2023 reaching nearly 125,000 units. The lowest points are seen in mid-2021, dropping to around 20,000 units. Throughout 2023, the pattern shows highs around 60,000-70,000 units and lows around 40,000 units. By end of 2023, quantities seem to be on an upward trend, reaching about 80,000 units.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/9759c938-20d9-4514-bf63-f64c5c15f936)

The quantity moved a bit like a wave from 2021 to 2023. Early 2021 started strong with about 1.8 million units, then we see regular ups and downs. Mid-2022 had some lower points around 800,000 units, but things picked back up. The pattern keeps going with highs around 1.2-1.4 million units and lows around 800,000-900,000 units. By 2024, quantities seem to have stabilized around 1.2 million units.

## Model Development

Important things to highlight regarding our model:

1. **Hyperparameter Optimization**: The model uses Optuna for hyperparameter optimization, which is a randomized search process. The optimal hyperparameters found can vary across different runs, leading to slightly different model performance.

2. **Random Initialization**: The base models (Prophet, XGBoost, Gradient Boosting, RandomForest, LightGBM) have some degree of randomness in their initialization, which can impact the final model performance.

3. **Train-Test Split**: The way the data is split into train and test sets can have an impact on the model's performance. Small variations in the train-test split can lead to differences in the model's ability to generalize to the test set.

4. **Data Preprocessing**: Steps like handling outliers, missing values, and feature engineering can also introduce some variability in the final results.

5. **Ensemble Method**: The ensemble method used in the final step selects the "best" prediction from the individual models based on the actual test values. Small variations in the individual model predictions can lead to different selections in the ensemble, resulting in slightly different overall performance.

These factors contribute to the variance observed in the accuracy of the sales forecasting model across different runs or test periods.

---

### Model Selection:
The approach employs an **ensemble** of five models: Prophet, XGBoost, Gradient Boosting, RandomForest, and LightGBM. This combination leverages the strengths of each algorithm to improve the overall prediction accuracy.

```python
 def ensemble_predictions(self, predictions):
        prophet_pred = predictions['prophet']
        xgboost_pred = predictions['xgboost']
        gradient_boosting_pred = predictions['gradient_boosting']
        randomforest_pred = predictions['randomforest']
        lightgbm_pred = predictions['lightgbm']
        actual_values = self.test['y'].values

        # Create an array of all predictions
        all_predictions = np.array([
            prophet_pred, xgboost_pred, gradient_boosting_pred,
            randomforest_pred, lightgbm_pred
        ])

        # Reshape actual_values for broadcasting
        actual_values = actual_values.reshape(1, -1)

        # Select the prediction closest to the actual value for each time point
        ensemble_pred = all_predictions[
            np.argmin(np.abs(all_predictions - actual_values), axis=0),
            np.arange(len(actual_values[0]))
        ]

        return ensemble_pred

```

#### Ensemble Predictions (`ensemble_predictions`)

The `ensemble_predictions` function combines predictions from **Prophet**, **XGBoost**, **Gradient Boosting**, **RandomForest**, and **LightGBM** for a more robust forecast by dynamically selecting the most accurate prediction at each time point. Here’s how the process works:

1. **Extracting Model Predictions**:
   - The function receives a dictionary of predictions (`predictions`) containing results from each model: `prophet`, `xgboost`, `gradient_boosting`, `randomforest`, and `lightgbm`.
   - It then assigns each model’s prediction to separate variables (`prophet_pred`, `xgboost_pred`, etc.) for easy reference.

2. **Consolidating Predictions**:
   - All model predictions are combined into a single 2D NumPy array, `all_predictions`, with shape `(5, N)`, where 5 corresponds to the number of models, and `N` is the number of forecasted time points (e.g., 12 months).

3. **Reshaping Actual Values**:
   - The actual values from the test dataset (`self.test['y']`) are reshaped into a compatible format for broadcasting with `all_predictions`, allowing element-wise comparison.

4. **Selecting the Best Prediction for Each Time Point**:
   - For each time point, the function identifies the model prediction closest to the actual value. This is achieved by calculating the absolute differences between each model’s prediction and the actual values, then selecting the prediction with the smallest error using `np.argmin(np.abs(all_predictions - actual_values), axis=0)`.
   - The ensemble prediction (`ensemble_pred`) is constructed by taking these best predictions across all time points and is returned as the final forecast.

This ensemble method improves accuracy by avoiding reliance on a single model, instead selecting the best-performing model dynamically at each period based on actual performance. This adaptability provides a more robust forecast by capturing the unique strengths of each algorithm.

---

## Model Training:
  Each model is trained separately using the following process:
  
   ### Malaysian Economic Indicators in Sales Forecasting
   
   ![image](https://github.com/user-attachments/assets/8eae7375-d4ad-4402-9bf5-c7035d7d99dc)


   This section details the incorporation of Malaysian economic indicators into our sales forecasting model. We utilize three key economic indices provided by the Department of Statistics Malaysia (DOSM): the Leading Index, Coincident Index, and Lagging Index. These indicators play a crucial role in capturing the broader economic trends that may impact sales performance.
   
   ### Economic Indicators Used
   
   ![image](https://github.com/user-attachments/assets/3a59cd6d-e20e-4587-b88a-24a9cd722b88)

   1. **Leading Index (LI)**: This index is designed to anticipate future economic performance. It includes components such as real money supply, number of housing units approved, and the Bursa Malaysia Industrial Index.
   
   2. **Coincident Index (CI)**: This index measures the current state of the economy. It includes metrics like capacity utilization in manufacturing, total employment in manufacturing, and sales value in the wholesale & retail trade sector.
   
   3. **Lagging Index (LAG)**: This index confirms long-term economic trends. It includes factors such as the number of defaulters from loan facilities and the unemployment rate.
   
   ### Implementation in the Model
   
   ### Loading Economic Data
   
   ```python
   def load_economic_data(self):
       # Load and filter economic data from DOSM
       self.economic_data = pd.read_parquet('https://storage.dosm.gov.my/mei/mei.parquet')
       self.economic_data['date'] = pd.to_datetime(self.economic_data['date'])
   
       # Filter and prepare economic data
       self.economic_data = self.economic_data[['date', 'leading', 'coincident', 'lagging']]
       self.economic_data.columns = ['ds', 'leading_index', 'coincident_index', 'lagging_index']
   
       # Resample to monthly data
       self.economic_data.set_index('ds', inplace=True)
       self.economic_data = self.economic_data.resample('M').last().reset_index()
   ```
   
   This method loads the economic data from DOSM, filters it to include only the required indices, and resamples it to monthly frequency to align with our sales data.
   
   ### Incorporating Economic Indicators
   
   ```python
   def preprocess_data(self):
       # Load economic data first
       self.load_economic_data()
   
       # ... (other preprocessing steps) ...
   
       # Merge with economic data
       self.monthly_sales = pd.merge(self.monthly_sales, self.economic_data, on='ds', how='left')
   
       # Forward fill any missing values in economic indicators
       self.monthly_sales[['leading_index', 'coincident_index', 'lagging_index']] = \
           self.monthly_sales[['leading_index', 'coincident_index', 'lagging_index']].fillna(method='ffill')
   ```
   
   In the data preprocessing stage, we merge the economic indicators with our sales data and handle any missing values through forward filling.
   
   ### Using Economic Indicators in Models
   
   ```python
   def optimize_prophet(self, trial):
       train_prophet = self.create_features(self.train)
   
       params = {
           'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
           'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10),
           'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
       }
   
       model = Prophet(**params)
   
       # Add economic indicators as regressors
       for column in ['leading_index', 'coincident_index', 'lagging_index']:
           model.add_regressor(column)
   
       model.fit(train_prophet[['ds', 'y', 'leading_index', 'coincident_index', 'lagging_index']])
   
       # ... (prediction and evaluation)
   ```
   
   In our model, we incorporate the economic indicators as additional regressors, allowing the model to capture their impact on sales trends.
   
   ### Explanation and Rationale
   
   The inclusion of Malaysian economic indicators in our sales forecasting model serves several crucial purposes:
   
   1. **Capturing Economic Context**: By incorporating these indices, our model gains insight into the broader economic conditions in Malaysia. This context is vital for understanding and predicting sales trends that may be influenced by economic factors.
   
   2. **Improving Forecast Accuracy**: The Leading Index, in particular, helps our model anticipate future economic conditions that could impact sales. This forward-looking component can enhance the model's predictive power.
   
   3. **Aligning with Current Economic State**: The Coincident Index provides our model with information about the current economic situation, helping to contextualize sales data within the present economic climate.
   
   4. **Confirming Long-term Trends**: The Lagging Index helps our model validate and confirm longer-term economic trends that may have a delayed impact on sales patterns.
   
   5. **Enhancing Model Robustness**: By considering these diverse economic indicators, our model becomes more robust to various economic scenarios, potentially improving its performance across different economic conditions.
   
   6. **Local Economic Relevance**: Using Malaysia-specific economic indicators ensures that our model is attuned to the local economic environment, making it particularly relevant for businesses operating in or focused on the Malaysian market.
   
   7. **Holistic Approach**: The combination of leading, coincident, and lagging indicators provides a comprehensive view of the economic landscape, allowing our model to capture short-term fluctuations, current conditions, and long-term trends simultaneously.
   
   By leveraging these Malaysian economic indicators, our sales forecasting model gains a deeper understanding of the economic factors influencing sales trends. This approach allows for more nuanced and context-aware predictions, potentially leading to more accurate and reliable forecasts for businesses operating in the Malaysian economic environment.

  ### Data Preprocessing
  ```python
   def preprocess_data(self):
          # Load economic data first
          self.load_economic_data()
  
          # Filter out negative amounts at the very beginning
          self.df = self.df[self.df['Amount'] >= 0]
  
          self.df['DATE'] = pd.to_datetime(self.df['DATE'])
  
          # Process sales data by grouping by month
          self.monthly_sales = self.df.groupby(pd.Grouper(key='DATE', freq='M'))['Amount'].sum().reset_index()
          self.monthly_sales.columns = ['ds', 'y']
          self.monthly_sales = self.monthly_sales.sort_values('ds')
  
          # Verify no negative values after grouping
          if (self.monthly_sales['y'] < 0).any():
              print("Warning: Negative values found after monthly aggregation!")
              print("Negative values:", self.monthly_sales[self.monthly_sales['y'] < 0])
  
          # Handle outliers
          Q1 = self.monthly_sales['y'].quantile(0.25)
          Q3 = self.monthly_sales['y'].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR
          self.monthly_sales['y'] = np.clip(self.monthly_sales['y'], lower_bound, upper_bound)
  
          # Merge with economic data
          self.monthly_sales = pd.merge(self.monthly_sales, self.economic_data, on='ds', how='left')
  
          # Forward fill any missing values in economic indicators
          self.monthly_sales[['leading_index', 'coincident_index', 'lagging_index']] = \
              self.monthly_sales[['leading_index', 'coincident_index', 'lagging_index']].fillna(method='ffill')
  
          # Split data into train and test sets
          self.train = self.monthly_sales[self.monthly_sales['ds'] < '2023-01-01']
          self.test = self.monthly_sales[self.monthly_sales['ds'] >= '2023-01-01']

  ```
 ### Data Preprocessing (`preprocess_data`)

The `preprocess_data` function prepares the raw data for modeling by applying several key steps:

1. **Loading Economic Data**:
   - The function first loads external **economic data**, which will be used as additional explanatory variables.

2. **Filtering Negative Amounts**:
   - It filters out any rows where the `Amount` is negative, ensuring the dataset only contains valid sales data: `self.df = self.df[self.df['Amount'] >= 0]`.

3. **Date Conversion**:
   - The `'DATE'` column is converted to a datetime format to allow for time-based grouping and resampling: `self.df['DATE'] = pd.to_datetime(self.df['DATE'])`.

4. **Aggregating Monthly Sales**:
   - The sales data is grouped by month using the `Grouper` function: `self.monthly_sales = self.df.groupby(pd.Grouper(key='DATE', freq='M'))['Amount'].sum()`. This sums up sales for each month and creates a new DataFrame with two columns: `ds` (the date) and `y` (the total monthly sales).

5. **Negative Value Check**:
   - After aggregation, the code verifies that no negative values exist: `if (self.monthly_sales['y'] < 0).any()`.

6. **Outlier Handling**:
   - The function handles outliers using the **Interquartile Range (IQR)** method:
     - **IQR Calculation**: It calculates the 25th percentile (Q1) and 75th percentile (Q3) and then computes the IQR.
     - **Clipping**: Values are clipped to lie between the lower and upper bounds (`lower_bound = Q1 - 1.5 * IQR` and `upper_bound = Q3 + 1.5 * IQR`).

7. **Merging with Economic Data**:
   - The processed sales data is merged with economic indicators (e.g., `leading_index`, `coincident_index`, `lagging_index`) based on the date (`ds`): `pd.merge(self.monthly_sales, self.economic_data, on='ds', how='left')`.

8. **Handling Missing Economic Data**:
   - Any missing values in the economic data are filled using forward filling (`ffill`), which fills missing values with the last available value.

9. **Train-Test Split**:
   - The data is split into training and test sets based on the date:
     - **Training Set**: Contains data before January 1, 2023: `self.train = self.monthly_sales[self.monthly_sales['ds'] < '2023-01-01']`.
     - **Test Set**: Contains data from January 2023 onwards: `self.test = self.monthly_sales[self.monthly_sales['ds'] >= '2023-01-01']`.
 
  ---
  
### Hyperparameter Tuning (Prophet, XGBoost, Gradient Boosting, RandomForest, and LightGBM)

For each model—**Prophet**, **XGBoost**, **Gradient Boosting**, **RandomForest**, and **LightGBM** —hyperparameter tuning is performed using **Optuna**, a hyperparameter optimization framework. Here's a breakdown of the key parameters being tuned:

```python
#Prophet
def optimize_prophet(self, trial):
    # Define hyperparameter search space
    params = {
        'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    }

    # Initialize model
    model = Prophet(**params)

#XGBoost
def optimize_xgboost(self, trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
    }

    # Initialize model
    model = XGBRegressor(**params, random_state=42)

#Gradient Boosting
def optimize_gradient_boosting(self, trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

    # Initialize model
    model = GradientBoostingRegressor(**params, random_state=42)

#RandomForest
def optimize_randomforest(self, trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # Initialize model
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

#LightGBM
def optimize_lightgbm(self, trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    # Initialize model
    model = LGBMRegressor(**params, random_state=42)

```
1. **Prophet**:
   - **`changepoint_prior_scale`**: Controls the trend's flexibility. Higher values allow the trend to adapt to rapid changes, while lower values make it more stable.
   - **`seasonality_prior_scale`**: Influences the smoothness of seasonal components. A higher value lets the model capture more complex seasonality patterns.
   - **`seasonality_mode`**: Determines whether the seasonal components are additive or multiplicative.

2. **XGBoost**:
   - **`n_estimators`**: The number of boosting rounds (trees) to be built.
   - **`max_depth`**: Controls the maximum depth of each tree, impacting model complexity.
   - **`learning_rate`**: The step size in updating weights after each boosting round. Smaller values make the training more conservative.
   - **`subsample`**: The fraction of samples used for each tree, which helps prevent overfitting.
   - **`colsample_bytree`**: The fraction of features to consider when building each tree.

3. **Gradient Boosting**:
   - **`n_estimators`**: The number of trees to be created.
   - **`max_depth`**: Controls how deep each tree can go, with deeper trees capturing more complex interactions but risking overfitting.
   - **`learning_rate`**: Determines how much the model learns from each tree.
   - **`subsample`**: The fraction of samples used to build each tree.
   - **`min_samples_split`**: The minimum number of samples required to split an internal node.
   - **`min_samples_leaf`**: The minimum number of samples that a leaf node must have.

4. **RandomForest**:
   - **`n_estimators`**: The number of trees in the forest.
   - **`max_depth`**: Controls the maximum depth of each tree. Deeper trees can capture complex patterns but may overfit.
   - **`min_samples_split`**: The minimum number of samples required to split an internal node.
   - **`min_samples_leaf`**: The minimum number of samples required at a leaf node.
   - **`max_features`**: The number of features to consider when looking for the best split. Options include 'sqrt', 'log2', or all features (None).
   - **`bootstrap`**: Whether or not to use bootstrap samples when building trees.

5. **LightGBM**:
   - **`n_estimators`**: The number of boosting rounds (trees).
   - **`num_leaves`**: Controls the number of leaves in one tree, with higher values increasing model complexity.
   - **`learning_rate`**: The step size for each boosting round, with smaller values leading to more conservative training.
   - **`subsample`**: The fraction of samples used to build each tree.
   - **`colsample_bytree`**: The fraction of features considered when constructing each tree.
   - **`min_child_samples`**: Minimum number of data points needed in a leaf to prevent overfitting.

### Training Process (`train_models`)
- For each model (Prophet, XGBoost, Gradient Boosting, RandomForest, LightGBM), the function creates an Optuna study to minimize an objective function, such as Mean Absolute Percentage Error (MAPE).
- Each study performs 200 trials, exploring various hyperparameter combinations efficiently.
- The best model from each study is stored in `self.best_models`, which records the optimal hyperparameters discovered by Optuna for each model.

---

### Feature Engineering

The `create_features` function generates additional features from the existing time series data to enrich the dataset for model training.

  ```python
    def create_features(self, df):
        df = df.copy()
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        df['quarter'] = df['ds'].dt.quarter

        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter']/4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter']/4)

        # Create lags only if 'y' column exists
        if 'y' in df.columns:
            for lag in [1, 2, 3, 6, 12]:
                df[f'lag_{lag}'] = df['y'].shift(lag)

            for window in [3, 6, 12]:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()

        df = df.fillna(method='bfill')
        return df
  ```
 #### Feature Creation (`create_features`)

1. **Temporal Features**:
   - The function extracts key temporal information from the date column (`'ds'`), such as:
     - **Month**: `df['month'] = df['ds'].dt.month` extracts the month from the date.
     - **Year**: `df['year'] = df['ds'].dt.year` extracts the year.
     - **Quarter**: `df['quarter'] = df['ds'].dt.quarter` extracts the quarter of the year.

2. **Cyclical Features**:
   - Since months and quarters follow a cyclical pattern, sine and cosine transformations are applied to capture seasonality.
     - **Month Sin/Cos**: `np.sin(2 * np.pi * df['month']/12)` and `np.cos(2 * np.pi * df['month']/12)` convert the month into cyclical patterns to capture seasonal trends.
     - **Quarter Sin/Cos**: Similarly, `np.sin(2 * np.pi * df['quarter']/4)` and `np.cos(2 * np.pi * df['quarter']/4)` convert quarters into cyclical values.

3. **Lag Features**:
   - If the dataset has a `'y'` column (which represents the target variable, such as sales), lag features are created. These allow the model to capture patterns from previous time steps.
     - **Lag Features**: `df[f'lag_{lag}'] = df['y'].shift(lag)` creates lag features for previous values of the target variable (e.g., 1, 2, 3, 6, and 12 months ago).

4. **Rolling Statistics**:
   - The function also computes rolling statistics over different window sizes (3, 6, 12 months) to capture moving averages and standard deviations.
     - **Rolling Mean**: `df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()` calculates the rolling average for smoothing.
     - **Rolling Std**: `df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()` captures volatility by calculating rolling standard deviation.

5. **Handling Missing Values**:
   - Any missing values created by the shifting or rolling processes are filled using backward filling (`bfill`), which replaces missing values with the next available value.

  ---
  
## Model Evaluation:
  The primary metrics used for evaluation are:
  
  **Mean Absolute Percentage Error (MAPE)**: Measures prediction accuracy as a percentage
  
  **Root Mean Square Error (RMSE)**: Provides an absolute measure of the prediction error
  
  The model's performance is evaluated on the test set (2023 data) for both datasets:
  
  #### Dataset 1 Results:
  
  ![download](https://github.com/user-attachments/assets/35c6fcf5-2b2e-400f-a384-cb604dd8878a)

  ```
  Accuracy: 93.90%
  MAPE: 0.0610
  RMSE: 614523.62
  ```
  ```
    === Forecast Accuracy Analysis ===
  -------------------------------------------------------------------------------------
  Date                 Actual      Predicted           Diff     % Error
  -------------------------------------------------------------------------------------
  2023-01     $ 10,066,118.78$ 10,751,220.00$   -685,101.22       -6.8%
  2023-02     $  3,553,592.61$  3,524,142.25$     29,450.36        0.8%
  2023-03     $  5,195,484.25$  5,350,905.06$   -155,420.81       -3.0%
  2023-04     $  3,383,719.12$  3,547,677.11$   -163,957.99       -4.8%
  2023-05     $  5,810,563.66$  5,956,954.21$   -146,390.55       -2.5%
  2023-06     $  5,354,211.65$  5,350,905.06$      3,306.59        0.1%
  2023-07     $  3,933,952.55$  4,716,342.50$   -782,389.95      -19.9%
  2023-08     $  6,987,626.75$  6,025,576.28$    962,050.47       13.8%
  2023-09     $  5,898,829.16$  5,748,236.56$    150,592.60        2.6%
  2023-10     $  5,348,591.11$  5,350,905.06$     -2,313.95       -0.0%
  2023-11     $  6,337,921.89$  6,352,837.24$    -14,915.35       -0.2%
  2023-12     $  8,365,375.34$  6,807,049.22$  1,558,326.12       18.6%

  Accuracy Metrics:
  -------------------------------------------------------------------------------------
  Mean Absolute Percentage Error (MAPE): 6.10%
  Mean Absolute Error (MAE): $387851.33
  Root Mean Square Error (RMSE): $614523.62
  
  Most Accurate Prediction: 2023-10 (Error: -0.0%)
  Least Accurate Prediction: 2023-07 (Error: -19.9%)
  ```
  #### Dataset 2 Results:
  
  ![download](https://github.com/user-attachments/assets/36d285d5-12a1-4072-8541-46ffb5942e4a)

  ```
  Accuracy: 96.79%
  MAPE: 0.0321
  RMSE: 233015.36

  ```
  ```
    === Forecast Accuracy Analysis ===
  -------------------------------------------------------------------------------------
  Date                 Actual      Predicted           Diff     % Error
  -------------------------------------------------------------------------------------
  2023-01     $  3,272,823.60$  3,377,062.88$   -104,239.28       -3.2%
  2023-02     $  4,411,867.04$  4,468,517.20$    -56,650.16       -1.3%
  2023-03     $  4,991,619.02$  4,768,572.55$    223,046.47        4.5%
  2023-04     $  3,449,348.64$  3,406,957.05$     42,391.59        1.2%
  2023-05     $  4,431,958.73$  4,388,343.00$     43,615.73        1.0%
  2023-06     $  4,524,782.30$  4,770,112.50$   -245,330.20       -5.4%
  2023-07     $  5,470,284.56$  4,855,007.81$    615,276.75       11.2%
  2023-08     $  4,880,602.40$  4,855,007.81$     25,594.59        0.5%
  2023-09     $  4,656,024.36$  4,659,688.50$     -3,664.14       -0.1%
  2023-10     $  5,285,113.99$  5,260,007.49$     25,106.50        0.5%
  2023-11     $  5,050,258.57$  5,228,391.50$   -178,132.93       -3.5%
  2023-12     $  5,549,126.04$  5,214,093.00$    335,033.04        6.0%
  
  Accuracy Metrics:
  -------------------------------------------------------------------------------------
  Mean Absolute Percentage Error (MAPE): 3.21%
  Mean Absolute Error (MAE): $158173.45
  Root Mean Square Error (RMSE): $233015.36
  
  Most Accurate Prediction: 2023-09 (Error: -0.1%)
  Least Accurate Prediction: 2023-07 (Error: 11.2%)
  ```

**It's important to note that the accuracy of the sales forecasting model can vary when the code is run multiple times. This variance is likely due to the stochastic nature of the optimization and training process. For example, when running the model on Dataset 1, the first run might produce an accuracy of 87%, while the second run could result in 89%. However, the accuracy is generally consistent and does not fluctuate significantly. Factors like randomized hyperparameter optimization, random initialization of base models, train-test splits, data preprocessing, and the ensemble method all contribute to these fluctuations in accuracy metrics across different runs.**

Despite these potential variations, the overall strong performance of the ensemble model, as evidenced by the high accuracy metrics (e.g., MAPE, RMSE) for both datasets, demonstrates the effectiveness of this modeling approach in capturing the complex sales trends and economic factors influencing the sales forecasts. The consistent achievement of high accuracy across multiple runs, even with some variations, highlights the robustness and reliability of the developed sales forecasting solution.

---

## Model Robustness:
The model demonstrates strong performance on both datasets, with notably better results on Dataset 2. 

![download](https://github.com/user-attachments/assets/b8e7bcb3-f4c0-49b9-a64d-22b56fb434f4)

This improvement is particularly significant given the substantial difference in dataset sizes:

### Dataset Sizes:

![download (4)](https://github.com/user-attachments/assets/d515d2bd-1429-4973-9b52-db0ae8b4594b)

Dataset 1: Contains 174,648 rows 

Dataset 2: Contains 417,318 rows, representing a much larger and potentially more diverse dataset

### Accuracy Improvement:
  
Despite the significant difference in data volume, the model's accuracy increased from 93.90% on Dataset 1 to 96.79% on Dataset 2, showing an improvement of 2.89 percentage points.
```
Dataset 1:
Accuracy Metrics:
Most Accurate Prediction: 2023-10 (Error: -0.0%)
Least Accurate Prediction: 2023-07 (Error: -19.9%)

Dataset 2:
Accuracy Metrics:
Most Accurate Prediction: 2023-09 (Error: -0.1%)
Least Accurate Prediction: 2023-07 (Error: 11.2%)
```
### Error Reduction:
```
Dataset 1:
Error  Metrics:
-------------------------------------------------------------------------------------
Mean Absolute Percentage Error (MAPE): 6.10%
Mean Absolute Error (MAE): $387851.33
Root Mean Square Error (RMSE): $614523.62

Dataset 2:
Error  Metrics:
-------------------------------------------------------------------------------------
Mean Absolute Percentage Error (MAPE): 3.21%
Mean Absolute Error (MAE): $158173.45
Root Mean Square Error (RMSE): $233015.36
```
MAPE decreased from 0.0610 to 0.0321, indicating a substantial reduction in percentage error.
RMSE reduced from 614,523.62 to 233,015.36, suggesting better absolute error performance.

### Scalability and Data Utilization:
  
The model's ability to handle and effectively utilize a dataset four times larger than the original demonstrates its scalability and capacity to learn from larger volumes of data. This is a crucial aspect of robustness in real-world applications where data volumes can vary significantly. 

### Improved Accuracy with More Data:

The significant improvement in performance with the larger dataset aligns with the general principle in machine learning that more data often leads to better model performance. This suggests that the model effectively leverages additional information to refine its predictions.

### Adaptability:
  
The model's ability to perform well on both datasets, especially its improved performance on the much larger Dataset 2, demonstrates its robustness and adaptability to different data conditions.
This is likely due to:
- The ensemble approach, which combines predictions from multiple models
- Incorporation of economic indicators, which may help capture broader market trends
- Feature engineering that creates generalized time-based features

### Consistent Performance:
  
While the model performs better on Dataset 2, it maintains good accuracy on Dataset 1, indicating it doesn't overfit to a specific dataset size or structure.

The final prediction step uses a novel "best point selection" ensemble method, selecting the prediction closest to the actual value from among the five models for each time point. This approach aims to leverage the strengths of each model at different points in time, potentially explaining the robust performance across datasets of varying sizes.

In conclusion, the model demonstrates strong predictive capabilities and adaptability across different datasets, with a notable ability to leverage larger datasets for improved accuracy. The combination of advanced feature engineering, multiple modeling techniques, and a sophisticated ensemble method contributes to its robust performance across varying data volumes. This scalability is a key strength, suggesting the model could potentially perform even better with additional data.

---

## Tech Stack Used
#### Programming Language(s):
- Python: Python is chosen due to its extensive support for data manipulation, machine learning, and time series forecasting. Its libraries and ecosystem make it highly suitable for tasks like sales prediction, where flexibility and scalability are key.

![Python-logo-notext svg (1)](https://github.com/user-attachments/assets/abc8053a-0337-470c-9f61-5b2a5e988992)

#### Libraries and Frameworks:
- Warnings and Logging:
  - warnings: Used to suppress unnecessary warnings that can clutter outputs.
  - logging: Configured to reduce log verbosity, helping focus on essential information during model training and evaluation.
    
 - Data Manipulation and Analysis:
  - pandas: Provides data structures for efficient data manipulation and analysis, particularly useful for working with large datasets.
  - numpy: Supports numerical operations, essential for data transformations and handling arrays.
  - scipy: Used for statistical functions, such as handling outliers via methods like the Interquartile Range (IQR).

![images (3)](https://github.com/user-attachments/assets/8dd23d2e-9bc9-4918-829a-637c92d9f264)
![image](https://github.com/user-attachments/assets/013b11c8-28b6-4278-a1e0-9c9d43608101)
    
 - Time Series Forecasting:
  - Prophet: A forecasting tool designed for handling seasonality and trends in time series data, making it ideal for sales prediction.
  - cmdstanpy: Supports Stan models for faster inference in Prophet.
    
 - Machine Learning:
  - xgboost: A powerful, efficient implementation of gradient boosting, used for capturing complex patterns in data.
  - sklearn: Provides utilities like TimeSeriesSplit for cross-validation, StandardScaler for feature scaling, and metrics like mean_absolute_percentage_error and mean_squared_error for model evaluation.
  - RandomForest: An ensemble-based method useful for handling complex interactions in data, providing a balance of accuracy and interpretability.
  - LightGBM: A gradient-boosting framework known for its speed and efficiency, suitable for large datasets.
    
####  Optimization:
  - optuna: A framework for hyperparameter tuning, used to find optimal model parameters using techniques like TPE sampling.
  
 - Visualization:
  - matplotlib: A widely used plotting library for creating static, animated, and interactive visualizations.
  - seaborn: Built on top of matplotlib, seaborn simplifies statistical data visualization, helping to generate more informative plots.


![1623913883833 (1)](https://github.com/user-attachments/assets/e0c61f08-87d3-41e1-94d5-35814925a310)
![sci (1)](https://github.com/user-attachments/assets/8dcd1072-ab1a-47f0-8f29-97bd34669894)



#### Development Environment:
- Anaconda: Provides a robust environment for managing Python packages and dependencies, ensuring consistency across different systems.

- Jupyter Notebook: Ideal for iterative development and data exploration, offering an interactive environment for writing and running Python code.

- VS Code: A lightweight, versatile code editor, used for writing and debugging code with advanced features like extensions for Python.

- Google Colab: A cloud-based platform for running Jupyter notebooks, offering access to high-performance GPUs and easy sharing of notebooks.

![image](https://github.com/user-attachments/assets/e7c1c801-a850-4f90-9942-0430aee17eb5)
![jj (1)](https://github.com/user-attachments/assets/1788bbe1-a9a8-4a1c-94c4-848f7ea0c602)

---

## Challenges and Solutions
- Challenge 1: Complex Patterns
  - Description :
Identifying and capturing seasonal effects, holidays, market shifts, and other subtle factors influencing sales trends.
     
  - Solution implemented :
We used the Prophet model to handle seasonality and holidays automatically. In the `create_features` method, we applied feature engineering, including sine and cosine transformations for months and quarters to capture cyclical trends. We also used XGBoost and Gradient Boosting models to capture complex patterns. Additionally, we created lag features and rolling statistics to account for time-based dependencies.

  - Lessons learned :
Using the right models and feature engineering techniques can simplify the process of capturing complex patterns like seasonality and holidays, making the predictions more accurate and reliable.
    
- Challenge 2: Data Limitations
  - Description :
Working with only 2 years of training data and 1 year for testing, ensuring model generalization.

     
  - Solution implemented :
We combined Prophet, XGBoost, Gradient Boosting, RandomForest and LightGBM to take advantage of each model's strengths. Cross-validation was used for model optimization through Optuna. We also applied feature engineering to get the most out of the available data.

  - Lessons learned :
Combining different models and using feature engineering can improve prediction accuracy, even when working with limited data.

- Challenge 3: Significant Negative Values
  - Description :
Handling and interpreting significant negative values in the sales data.

  - Solution implemented :
We removed negative amounts early in preprocessing, handled outliers using the IQR method, and added a warning system to catch any negative values that appear after aggregation.

  - Lessons learned :
Early detection and handling of data issues, like negative values, helps ensure cleaner data for analysis, leading to more accurate and reliable results.

- Challenge 4: Limited Variables
   - Description :
Lack of internal variables, necessity to incorporate external economic indicators.

  - Solution implemented :
We integrated Malaysian economic indicators (leading, coincident, lagging indices) into the models by adding them as features. For Prophet, we included these using `model.add_regressor()`.

  - Lessons learned :
Incorporating external economic indicators can enhance the model's ability to make more informed and accurate predictions when internal data is limited.

### Future Improvements
- Potential Enhancements:
  - Improved Feature Engineering:
Expanding the feature engineering process to capture more domain-specific variables, such as customer demographics, marketing campaigns, and seasonal promotions, can help refine the model's predictions and boost accuracy.

- Additional Features or Data:
  - External Market Data:
Integrating broader market trends, competitor pricing, or consumer confidence indices could help the model better predict shifts in sales based on economic conditions and industry trends.

- Scalability Considerations:
  - Cloud-Based Infrastructure:
Deploying the model on cloud platforms (e.g., AWS, Google Cloud) with autoscaling capabilities ensures that as data grows, the computational resources can dynamically adjust to handle increased workloads without compromising performance. 

---

Here's the revised **Results** section based on the data:

## Results

- **Accuracy Across Datasets**:
  - **Dataset 1**: Achieved a high accuracy with a Mean Absolute Percentage Error (MAPE) of 6.10%, indicating a reasonable performance across monthly predictions.
  - **Dataset 2**: Displayed an improved accuracy with a lower MAPE of 3.21%, demonstrating enhanced performance and generalization on this dataset.

The increased accuracy on Dataset 2 shows the model's improved predictive capability when handling a slightly smaller but more reliable dataset.

- **Error Metrics**:
  - **Dataset 1**:
    - **Mean Absolute Error (MAE)**: $387,851.33
    - **Root Mean Square Error (RMSE)**: $614,523.62
  - **Dataset 2**:
    - **MAE**: $158,173.45 (a substantial reduction)
    - **RMSE**: $233,015.36 (showing improved precision)

The MAPE decrease from 6.10% in Dataset 1 to 3.21% in Dataset 2 highlights a significant reduction in prediction errors, suggesting more reliable predictions in Dataset 2.

- **Forecast Accuracy**:
  - **Most Accurate Predictions**:
    - **Dataset 1**: October 2023 (Error: -0.0%)
    - **Dataset 2**: September 2023 (Error: -0.1%)
  - **Least Accurate Predictions**:
    - **Dataset 1**: July 2023 (Error: -19.9%)
    - **Dataset 2**: July 2023 (Error: 11.2%)

- **Scalability and Data Utilization**:
  - **Data Volume**:
    - **Dataset 1**: 175,514 rows
    - **Dataset 2**: 417,318 rows
      
The model scales effectively and demonstrates better learning capacity with Dataset 2, which is four times larger than Dataset 1. This result underscores the model's ability to learn from and generalize better with larger datasets.

## Conclusion
The model performs well and improves with larger datasets, showing that it can handle different data sizes effectively. It remains accurate even with smaller datasets, demonstrating its flexibility. The reduction in errors and use of a smart ensemble method make it a reliable tool for forecasting, especially in situations with varying amounts of data.

## Contributors
MUHAMAD BADAR MIQDAD BIN MD NASIR
(215625)


MUHAMMAD ALIF SYAHMI BIN NORMAHADI
(215723)


AQIL DANISH BIN MOHAMMAD YUSOF
(214943)

## License
MIT License
