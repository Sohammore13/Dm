"""
Experiment 1: Design and Implement a Star Schema using SQL Server/DataGrip
Objective: Create Fact and Dimension tables, define primary/foreign keys, and execute analytical queries.

================================================================================
-- NATIVE SQL IMPLEMENTATION (SQL Server / DataGrip / PostgreSQL Compatible)
================================================================================
-- 1. Create Dimension Tables
CREATE TABLE DimDate (
    DateKey INT PRIMARY KEY,
    FullDate DATE NOT NULL,
    DayOfWeek VARCHAR(10),
    MonthName VARCHAR(10),
    Quarter INT,
    Year INT
);

CREATE TABLE DimCustomer (
    CustomerKey INT PRIMARY KEY,
    CustomerID VARCHAR(10) NOT NULL,
    CustomerName VARCHAR(100),
    Gender VARCHAR(10),
    City VARCHAR(50),
    State VARCHAR(50),
    Country VARCHAR(50)
);

CREATE TABLE DimProduct (
    ProductKey INT PRIMARY KEY,
    ProductID VARCHAR(10) NOT NULL,
    ProductName VARCHAR(100),
    Category VARCHAR(50),
    SubCategory VARCHAR(50),
    UnitPrice DECIMAL(10, 2)
);

CREATE TABLE DimStore (
    StoreKey INT PRIMARY KEY,
    StoreID VARCHAR(10) NOT NULL,
    StoreName VARCHAR(100),
    Region VARCHAR(50),
    Country VARCHAR(50)
);

-- 2. Create Fact Table with Foreign Key Constraints
CREATE TABLE FactSales (
    SalesKey INT PRIMARY KEY,
    DateKey INT NOT NULL,
    CustomerKey INT NOT NULL,
    ProductKey INT NOT NULL,
    StoreKey INT NOT NULL,
    QuantitySold INT,
    TotalRevenue DECIMAL(12, 2),
    TotalCost DECIMAL(12, 2),
    Profit DECIMAL(12, 2),
    CONSTRAINT FK_FactSales_Date FOREIGN KEY (DateKey) REFERENCES DimDate(DateKey),
    CONSTRAINT FK_FactSales_Customer FOREIGN KEY (CustomerKey) REFERENCES DimCustomer(CustomerKey),
    CONSTRAINT FK_FactSales_Product FOREIGN KEY (ProductKey) REFERENCES DimProduct(ProductKey),
    CONSTRAINT FK_FactSales_Store FOREIGN KEY (StoreKey) REFERENCES DimStore(StoreKey)
);

-- 3. Star Schema Analytical Query (Join Fact with Dimensions)
SELECT 
    d.Year,
    d.Quarter,
    p.Category,
    s.Region,
    SUM(f.QuantitySold) AS TotalUnitsSold,
    SUM(f.TotalRevenue) AS TotalRevenue,
    SUM(f.Profit) AS TotalProfit
FROM FactSales f
JOIN DimDate d ON f.DateKey = d.DateKey
JOIN DimProduct p ON f.ProductKey = p.ProductKey
JOIN DimStore s ON f.StoreKey = s.StoreKey
JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
GROUP BY d.Year, d.Quarter, p.Category, s.Region
ORDER BY d.Year, TotalRevenue DESC;
================================================================================
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import sqlite3
import pandas as pd
import numpy as np

def load_source_data():
    """Load dataset from library, with robust offline fallback."""
    try:
        import seaborn as sns
        return sns.load_dataset('tips')
    except Exception:
        # Standard built-in dataset simulation for offline lab environments
        np.random.seed(42)
        n = 244
        days = np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], size=n, p=[0.25, 0.15, 0.35, 0.25])
        times = np.where(np.isin(days, ['Thur', 'Fri']), np.random.choice(['Lunch', 'Dinner'], size=n), 'Dinner')
        sexes = np.random.choice(['Female', 'Male'], size=n)
        smokers = np.random.choice(['No', 'Yes'], size=n, p=[0.6, 0.4])
        sizes = np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.6, 0.15, 0.15, 0.03, 0.02])
        bills = np.round(np.random.exponential(scale=18, size=n) + 5, 2)
        tips = np.round(bills * np.random.uniform(0.10, 0.22, size=n), 2)
        return pd.DataFrame({
            'total_bill': bills, 'tip': tips, 'sex': sexes,
            'smoker': smokers, 'day': days, 'time': times, 'size': sizes
        })

def main():
    print("=" * 70)
    print("Experiment 1: Star Schema Implementation & Verification")
    print("=" * 70)

    # Getting dataset from library itself
    tips_df = load_source_data()
    print(f"[+] Loaded source dataset: {tips_df.shape[0]} records")

    # In-memory SQLite relational database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 1. Create Dimension & Fact Tables in SQLite
    cursor.executescript("""
    CREATE TABLE DimDate (
        DateKey INTEGER PRIMARY KEY AUTOINCREMENT,
        DayName TEXT,
        TimeOfDay TEXT
    );

    CREATE TABLE DimCustomer (
        CustomerKey INTEGER PRIMARY KEY AUTOINCREMENT,
        Sex TEXT,
        Smoker TEXT
    );

    CREATE TABLE DimParty (
        PartyKey INTEGER PRIMARY KEY AUTOINCREMENT,
        PartySize INTEGER
    );

    CREATE TABLE FactRestaurantSales (
        SalesKey INTEGER PRIMARY KEY AUTOINCREMENT,
        DateKey INTEGER,
        CustomerKey INTEGER,
        PartyKey INTEGER,
        TotalBill REAL,
        Tip REAL,
        TipPercentage REAL,
        FOREIGN KEY (DateKey) REFERENCES DimDate(DateKey),
        FOREIGN KEY (CustomerKey) REFERENCES DimCustomer(CustomerKey),
        FOREIGN KEY (PartyKey) REFERENCES DimParty(PartyKey)
    );
    """)

    # 2. Populate Dimensions
    unique_dates = tips_df[['day', 'time']].drop_duplicates().reset_index(drop=True)
    for _, row in unique_dates.iterrows():
        cursor.execute("INSERT INTO DimDate (DayName, TimeOfDay) VALUES (?, ?)", (str(row['day']), str(row['time'])))

    unique_cust = tips_df[['sex', 'smoker']].drop_duplicates().reset_index(drop=True)
    for _, row in unique_cust.iterrows():
        cursor.execute("INSERT INTO DimCustomer (Sex, Smoker) VALUES (?, ?)", (str(row['sex']), str(row['smoker'])))

    unique_size = tips_df[['size']].drop_duplicates().reset_index(drop=True)
    for _, row in unique_size.iterrows():
        cursor.execute("INSERT INTO DimParty (PartySize) VALUES (?)", (int(row['size']),))

    conn.commit()

    # Load Dimension IDs for Fact Table mapping
    dim_date_df = pd.read_sql("SELECT * FROM DimDate", conn)
    dim_cust_df = pd.read_sql("SELECT * FROM DimCustomer", conn)
    dim_party_df = pd.read_sql("SELECT * FROM DimParty", conn)

    # 3. Populate Fact Table
    fact_records = []
    for _, row in tips_df.iterrows():
        d_key = dim_date_df[(dim_date_df['DayName'] == str(row['day'])) & (dim_date_df['TimeOfDay'] == str(row['time']))]['DateKey'].values[0]
        c_key = dim_cust_df[(dim_cust_df['Sex'] == str(row['sex'])) & (dim_cust_df['Smoker'] == str(row['smoker']))]['CustomerKey'].values[0]
        p_key = dim_party_df[dim_party_df['PartySize'] == int(row['size'])]['PartyKey'].values[0]
        tip_pct = round((row['tip'] / row['total_bill']) * 100, 2)
        fact_records.append((int(d_key), int(c_key), int(p_key), float(row['total_bill']), float(row['tip']), float(tip_pct)))

    cursor.executemany("""
    INSERT INTO FactRestaurantSales (DateKey, CustomerKey, PartyKey, TotalBill, Tip, TipPercentage)
    VALUES (?, ?, ?, ?, ?, ?)
    """, fact_records)
    conn.commit()

    print("[+] Successfully populated Star Schema (3 Dimensions + 1 Fact Table).")

    # 4. Analytical Query joining Fact with Dimensions
    query = """
    SELECT 
        d.DayName AS Day,
        d.TimeOfDay AS MealTime,
        c.Sex AS Gender,
        COUNT(f.SalesKey) AS TotalTransactions,
        ROUND(SUM(f.TotalBill), 2) AS GrossRevenue,
        ROUND(AVG(f.TipPercentage), 2) AS AvgTipPercentage
    FROM FactRestaurantSales f
    JOIN DimDate d ON f.DateKey = d.DateKey
    JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
    JOIN DimParty p ON f.PartyKey = p.PartyKey
    GROUP BY d.DayName, d.TimeOfDay, c.Sex
    ORDER BY GrossRevenue DESC;
    """

    result_df = pd.read_sql(query, conn)
    print("\n--- Star Schema Analytical Aggregation Query Result ---")
    print(result_df)

    conn.close()

if __name__ == "__main__":
    main()
