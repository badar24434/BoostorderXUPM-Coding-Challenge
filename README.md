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
- Dataset 1

![image](https://github.com/user-attachments/assets/135d0fbc-536f-45a6-9c32-8152829521e1)

The overall trend shows growth from 2021 to 2023. Starting around 6 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in early 2022 reaching around 14 million, and late 2023 showing similar highs. Despite some dips, each year tends to end at a higher point than it started. This indicates a positive growth trajectory for the business over the observed period.

- Dataset 2

![image](https://github.com/user-attachments/assets/a0e55358-f87d-4021-a314-268d60783897)

The overall trend shows growth from 2021 to 2023. Starting around 4 million in early 2021, the amounts fluctuate but generally increase over time. Notable peaks appear in mid-2022 reaching 5.5 million, and late 2023 showing similar highs. A the end of 2023 maintains these higher levels around 5-5.5 million. Despite some dips, each year tends to end at a higher point than it started.

### Analyze what day of the week has the highest sales
- Datset 1

  ![image](https://github.com/user-attachments/assets/b4cc48c9-6ec6-48e0-84e6-354d4887bd8f)

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

Order patterns are very consistent across months. Weekdays, especially Monday through Wednesday, regularly see 1,000-1,500 orders. January shows the highest spike with Monday hitting nearly 1,500 orders. Weekends drop significantly - Saturdays average around 100 orders and Sundays even less. The pattern is remarkably stable across all months, with slight variations in peak days.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/01977e3f-cdbb-469c-8216-3c4d811b69f1)

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

Order patterns are very consistent across months. Weekdays, especially Monday through Wednesday, regularly see 4,000-5,000 orders. June shows the highest spike with Tuesday hitting nearly 6,000 orders. Weekends drop significantly - Saturdays average around 1,000 orders and Sundays even less. The pattern is remarkably stable across all months, with slight variations in peak days.

### Total sales by month for all 2021-2023 transactions
- Dataset 1

  ![image](https://github.com/user-attachments/assets/d5d005ff-0350-4a6b-8dfb-4dfe72687847)

Sales follow a distinct yearly pattern. They start highest in January at about 34 million, then drop sharply to around 16 million in February. The trend then shows a gradual decrease until May, reaching the lowest point of about 12.5 million. From there, sales slowly increase through the rest of the year. The growth accelerates from September onwards, with a particularly steep rise from October to December. The year ends with December sales at about 22.5 million, though still below the January peak.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/4845eed7-a8e3-46ab-8fd8-1afe44d1d8cd)

Sales follow a clear yearly pattern. They start low in January (around 10.5 million) and build up through March (13 million). There's usually a dip in April, but things really pick up from September onwards. The end of the year is strongest - October through December show steady growth, reaching peak sales of about 14.2 million in December.


### Total Revenue by Month and Day of the Week
- Dataset 1

  ![image](https://github.com/user-attachments/assets/48d24c65-28b1-4b73-9752-54e10e59a5e6)

Weekdays generate most of the revenue throughout the year. Monday through Friday typically bring in between 2-5 million in revenue. There's significant variation, but Wednesday and Friday often show higher revenue, especially towards the end of the year. Weekends consistently show the lowest revenue, with Saturday and Sunday rarely exceeding 1 million. There's a notable peak in January for most weekdays, with Monday, Tuesday, and Wednesday all reaching around 5-6 million. December also shows high revenue for several weekdays, particularly Friday which peaks at about 6 million.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/87ad16bb-30d3-4e93-b4fc-ead299db9806)

Weekdays clearly drive most revenue throughout the year. Monday through Thursday typically bring in between 2.5-3.5 million in revenue. Wednesday and Thursday often lead the pack, especially in later months. Weekends tell a different story - Saturday and Sunday consistently show the lowest revenue, rarely going above 500,000. There's a noticeable peak in March for Monday and another in April for Wednesday, both hitting around 3.5 million.



### Total Quantity trend (by month) from 2021 to 2023
- Dataset 1

  ![image](https://github.com/user-attachments/assets/b9c8a26f-1647-4b90-8b3b-8ac11826c82f)

The quantity shows significant fluctuations from 2021 to 2023. Early 2021 started around 70,000 units, then we see regular ups and downs. There are two notable peaks: one in early 2022 at about 120,000 units, and another in early 2023 reaching nearly 125,000 units. The lowest points are seen in mid-2021, dropping to around 20,000 units. Throughout 2023, the pattern shows highs around 60,000-70,000 units and lows around 40,000 units. By end of 2023, quantities seem to be on an upward trend, reaching about 80,000 units.

- Dataset 2

  ![image](https://github.com/user-attachments/assets/9759c938-20d9-4514-bf63-f64c5c15f936)

The quantity moved a bit like a wave from 2021 to 2023. Early 2021 started strong with about 1.8 million units, then we see regular ups and downs. Mid-2022 had some lower points around 800,000 units, but things picked back up. The pattern keeps going with highs around 1.2-1.4 million units and lows around 800,000-900,000 units. By 2024, quantities seem to have stabilized around 1.2 million units.

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
