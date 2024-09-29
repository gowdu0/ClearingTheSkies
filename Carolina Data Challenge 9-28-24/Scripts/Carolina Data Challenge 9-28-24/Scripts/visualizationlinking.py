import pandas as pd
import plotly.express as px

csv_file = 'Data/merged_worldbank_demographic_data.csv'
df = pd.read_csv(csv_file)

df.rename(columns={'Aviation Emissions (Tons)': 'Emissions'}, inplace=True)

numeric_columns = ['GDP (current US$)', 'Population(Thousands)', 'CO2 Emissions (Metric Tons)', 'Emissions', 'Climate Vulnerability Score']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna()

vulnerability_threshold = 0.5
df_clean['Vulnerability Group'] = df_clean['Climate Vulnerability Score'].apply(
    lambda x: 'High Vulnerability' if x > vulnerability_threshold else 'Low Vulnerability'
)

fig_compare = px.scatter(
    df_clean, 
    x='Climate Vulnerability Score', 
    y='Emissions', 
    color='Vulnerability Group',
    labels={
        'Emissions': 'Aviation Emissions (Tons)',
        'Climate Vulnerability Score': 'Climate Vulnerability Score',
        'Vulnerability Group': 'Vulnerability Category'
    },
    title='Aviation Emissions vs. Climate Vulnerability (High vs. Low Vulnerability Countries)',
    trendline='ols',
    color_discrete_map={'High Vulnerability': 'red', 'Low Vulnerability': 'blue'}
)

fig_compare.update_layout(
    xaxis_title="Climate Vulnerability Score",
    yaxis_title="Aviation Emissions (Tons)",
    legend_title="Vulnerability Group",
    font=dict(size=12)
)

fig_compare.show()

print("\nExplanation:")
print("This scatter plot compares the Aviation Emissions (in Tons) against the Climate Vulnerability Score for high and low-vulnerability countries.")
print("Countries with a Climate Vulnerability Score > 0.5 are categorized as High Vulnerability (colored red), and those <= 0.5 as Low Vulnerability (colored blue).")
print("Regression lines help show the overall trend between emissions and vulnerability within each group.")
