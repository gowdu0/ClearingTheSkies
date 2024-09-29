import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

csv_file = 'Data/merged_worldbank_demographic_data.csv'
df = pd.read_csv(csv_file)

df.rename(columns={'Aviation Emissions (Tons)': 'Emissions'}, inplace=True)

numeric_columns = ['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)', 'Emissions', 'Climate Vulnerability Score']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df[['Year', 'Month']].astype(str).agg('-'.join, axis=1), format='%Y-%m')

df_clean = df.dropna()

low_vulnerability_threshold = 0.5
df_low_vulnerability = df_clean[df_clean['Climate Vulnerability Score'] <= low_vulnerability_threshold]

print("\nSummary Statistics for Emissions (Low Vulnerability Countries):")
print(df_low_vulnerability[['Emissions']].describe())

X = df_low_vulnerability[['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)']]
y = df_low_vulnerability['Emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_results = X_test.copy()
test_results['Actual Emissions'] = y_test
test_results['Predicted Emissions'] = y_pred
test_results['Date'] = df_low_vulnerability.loc[y_test.index, 'Date']

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
    title='Actual vs. Predicted Aviation Emissions Over Time (Low-Vulnerability Countries)',
    color_discrete_sequence=['blue', 'red']
)
fig_emissions.update_layout(legend_title_text='Emissions')
fig_emissions.show()

print("\nExplanation:")
print("This plot represents the Actual vs. Predicted Aviation Emissions for countries with low climate vulnerability.")
print("The dataset is filtered for countries with a Climate Vulnerability Score less than or equal to 0.5.")
