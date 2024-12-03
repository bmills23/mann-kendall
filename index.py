import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymannkendall import original_test
from scipy.stats import pearsonr

# Load and preprocess data
file_path = "data copy.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Debug: Print raw dataset information
print("Raw Dataset:")
print(data.head())
print(data.info())

# Select relevant columns
relevant_columns = ['Well ID', 'Date', 'Benzene', 'Toluene', 'Ethyl-Benzene', 
                    'Xylenes', 'TVPH', 'Water Table Elevation']
data = data[relevant_columns]

# Debug: Check dataset after column selection
print("\nAfter Selecting Relevant Columns:")
print(data.head())
print(data.isnull().sum())

# Parse Date column dynamically to handle multiple formats
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Debug: Check parsed dates
print("\nAfter Parsing Date:")
print(data.head())
print("Invalid dates:", data['Date'].isnull().sum())

# Convert numeric columns to appropriate data types
numeric_columns = ['Benzene', 'Toluene', 'Ethyl-Benzene', 'Xylenes', 'TVPH', 'Water Table Elevation']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Debug: Check data types after conversion
print("\nData Types After Conversion:")
print(data.dtypes)

# Drop rows with missing critical data
data = data.dropna(subset=['Date', 'Well ID'])

# Debug: Check dataset after dropping rows
print("\nAfter Dropping Rows with Missing Date or Well ID:")
print(data.head())
print("Remaining rows:", len(data))

# Analyze by Well ID
wells = data['Well ID'].unique()

# Debug: Print unique wells
print("\nUnique Wells:", wells)

# Analysis for each well
for well in wells:
    well_data = data[data['Well ID'] == well]

    # Ensure dataset for the well is valid
    if well_data.empty:
        print(f"No valid data for Well ID: {well}. Skipping analysis.")
        continue

    # Check if there are enough valid data points for the Mann-Kendall test
    well_data = well_data.dropna(subset=['Date', 'Benzene', 'Toluene', 'Ethyl-Benzene', 'Xylenes', 'TVPH'])
    if len(well_data) < 2:  # Mann-Kendall requires at least two data points
        print(f"Not enough valid data for Mann-Kendall test for Well ID: {well}. Skipping.")
        continue

    # Mann-Kendall Trend Analysis for all Constituents
    print(f"\n--- Analysis for Well ID: {well} ---")
    for constituent in ['Benzene', 'Toluene', 'Ethyl-Benzene', 'Xylenes', 'TVPH']:
        # Perform the Mann-Kendall test
        trend_result = original_test(well_data[constituent])
        print(f"Mann-Kendall Trend Test for {constituent}:\n")
        print(trend_result, '\n')
        
        # Explanation of results
        print(f"Explanation for {constituent}:\n")
        print(f"Trend: {trend_result.trend}\n")  # The trend direction (increasing, decreasing, or no trend)

        # Adjust wording if there is no trend
        if trend_result.trend == "no trend":
            trend_text = "significant trend"
        else:
            trend_text = f"{trend_result.trend} trend"
        
        # Check significance of p-value
        if trend_result.p < 0.05:
            print(f"p-value ({trend_result.p:.4f}) is significant, indicating a {trend_text}.\n")
        else:
            print(f"p-value ({trend_result.p:.4f}) is not significant, no evidence for a {trend_text}.\n")

        # Check significance of z-value
        if trend_result.z > 1.960 or trend_result.z < -1.960:
            print(f"z-value ({trend_result.z:.4f}) is significant, supporting the {trend_text}.\n")
        else:
            print(f"z-value ({trend_result.z:.4f}) is not significant, no strong evidence for a {trend_text}.\n")

        # Categorize strength of trend based on Tau
        if trend_result.Tau == 0:
            print("No association (Tau = 0).\n")
        elif 0 < trend_result.Tau <= 0.1:
            print("Very weak positive trend (Tau > 0 and ≤ 0.1).\n")
        elif 0.1 < trend_result.Tau <= 0.3:
            print("Weak positive trend (Tau > 0.1 and ≤ 0.3).\n")
        elif 0.3 < trend_result.Tau <= 0.5:
            print("Moderate positive trend (Tau > 0.3 and ≤ 0.5).\n")
        elif trend_result.Tau > 0.5:
            print("Strong positive trend (Tau > 0.5).\n")
        elif -0.1 <= trend_result.Tau < 0:
            print("Very weak negative trend (Tau ≥ -0.1 and < 0).\n")
        elif -0.3 <= trend_result.Tau < -0.1:
            print("Weak negative trend (Tau ≥ -0.3 and < -0.1).\n")
        elif -0.5 <= trend_result.Tau < -0.3:
            print("Moderate negative trend (Tau ≥ -0.5 and < -0.3).\n")
        elif trend_result.Tau < -0.5:
            print("Strong negative trend (Tau < -0.5).\n")

        # Provide additional context for S and variance (var_s)
        print(f"S-value: {trend_result.s}, which reflects the difference between positive and negative ranks.\n")
        print(f"Variance of S (var_s): {trend_result.var_s:.4f}, used for calculating z-value.\n")
        print(f"Estimated slope: {trend_result.slope:.6f} per time unit, indicating the rate of change.\n")
        print(f"Intercept: {trend_result.intercept:.6f}, which helps form the line of best fit for the trend.\n")

    # Correlation between Benzene and Water Table Elevation
    # Align Benzene and Water Table Elevation by dropping rows where either column has NaN
    valid_data = well_data[['Benzene', 'Water Table Elevation']].dropna()
    if valid_data.empty:
        print(f"No valid data for correlation analysis for Well ID: {well}. Skipping.")
    else:
        corr, p_value = pearsonr(valid_data['Water Table Elevation'], valid_data['Benzene'])
        print("Correlation (Benzene vs Water Table Elevation):", corr, "P-value:", p_value)

    # Plotting Benzene, Toluene, Ethyl-Benzene, Xylenes, and TVPH on the same graph
    plt.figure(figsize=(10, 6))

    # Plot each constituent
    plt.plot(well_data['Date'], well_data['Benzene'], label='Benzene', color='b', marker='o')
    plt.plot(well_data['Date'], well_data['Toluene'], label='Toluene', color='g', marker='x')
    plt.plot(well_data['Date'], well_data['Ethyl-Benzene'], label='Ethyl-Benzene', color='r', marker='s')
    plt.plot(well_data['Date'], well_data['Xylenes'], label='Xylenes', color='purple', marker='^')
    plt.plot(well_data['Date'], well_data['TVPH'], label='TVPH', color='orange', marker='d')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Concentration')
    plt.title(f'Concentration Trends for BTEX and TVPH - Well {well}')  # Corrected to use `well`
    plt.legend()

    # Rotate date labels for better visibility
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    plt.show()