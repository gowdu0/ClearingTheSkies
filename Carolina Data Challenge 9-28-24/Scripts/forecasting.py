import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

csv_file = 'Data/merged_worldbank_demographic_data.csv'
df = pd.read_csv(csv_file)

df.rename(columns={'Aviation Emissions (Tons)': 'Emissions'}, inplace=True)

numeric_columns = ['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)', 'Emissions']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

try:
    df['Date'] = pd.to_datetime(df['Year_Month'], format='%Y-%b')
except:
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].astype(str).agg('-'.join, axis=1), format='%Y-%m')

df_clean = df.dropna()

correlation_data = df_clean[['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)', 'Emissions']]
corr_matrix = correlation_data.corr(method='pearson')
print("\nPearson Correlation Coefficient Matrix:")
print(corr_matrix)

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='Viridis',
    title='Correlation Matrix Heatmap'
)
fig_corr.update_layout(coloraxis_showscale=True)
fig_corr.show()

X = df_clean[['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)']]
y = df_clean['Emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_results = X_test.copy()
test_results['Actual Emissions'] = y_test
test_results['Predicted Emissions'] = y_pred
test_results['Date'] = df_clean.loc[y_test.index, 'Date']

test_results.sort_values('Date', inplace=True)

aggregated_results = test_results.groupby('Date').agg({
    'Actual Emissions': 'sum',
    'Predicted Emissions': 'sum'
}).reset_index()

fig_emissions = px.line(
    aggregated_results,
    x='Date',
    y=['Actual Emissions', 'Predicted Emissions'],
    labels={'value': 'Aviation Emissions (Tons)', 'Date': 'Date', 'variable': 'Legend'},
    title='Actual vs. Predicted Aviation Emissions Over Time (Aggregated Across All Countries)',
    color_discrete_sequence=['blue', 'red']
)
fig_emissions.update_layout(legend_title_text='Emissions')
fig_emissions.show()

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

print("\nExplanation:")
print("The 'Emissions' column represents the aviation emissions (in Tons) for each country on a monthly basis.")
print("The model is trained using data from all countries across all months in the dataset.")
print("The final visualization shows the aggregated Actual and Predicted Aviation Emissions over time for the entire dataset.")
