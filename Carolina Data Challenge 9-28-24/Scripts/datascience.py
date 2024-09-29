import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text
import numpy as np

df = pd.read_excel('Data/emissions.xlsx')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

emissions_columns = df.columns.drop(['Country', 'Climate Vulnerability Score'])
emissions_long = df.melt(id_vars=['Country', 'Climate Vulnerability Score'], value_vars=emissions_columns, var_name='Date', value_name='Emissions')
emissions_long['Date'] = pd.to_datetime(emissions_long['Date'], format='%Y-%b')
emissions_long['Emissions'] = pd.to_numeric(emissions_long['Emissions'], errors='coerce').fillna(0)
emissions_long['Emissions (Trillions of Tonnes)'] = emissions_long['Emissions'] / 1_000_000_000

total_emissions = emissions_long.groupby('Country')['Emissions (Trillions of Tonnes)'].sum().reset_index()
df_merged = pd.merge(total_emissions, df[['Country', 'Climate Vulnerability Score']].drop_duplicates(), on='Country')

high_emission_countries = ['Australia', 'Canada', 'France', 'Germany', 'Japan', 'Netherlands', 'Spain', 'United Kingdom', 'United States', 'United Arab Emirates']
high_vulnerability_countries = ['Bangladesh', 'Haiti', 'Madagascar', 'Maldives', 'Mozambique', 'Nepal', 'Solomon Islands', 'Vanuatu']

def categorize_country(country):
    if country in high_emission_countries:
        return 'High Emission'
    elif country in high_vulnerability_countries:
        return 'High Vulnerability'
    else:
        return 'Other'

df_merged['Country Group'] = df_merged['Country'].apply(categorize_country)

plt.figure(figsize=(16, 12))
sns.set(style="whitegrid")
palette = {'High Emission': 'red', 'High Vulnerability': 'blue', 'Other': 'green'}
scatter = sns.scatterplot(data=df_merged, x='Emissions (Trillions of Tonnes)', y='Climate Vulnerability Score', hue='Country Group', palette=palette, s=150, edgecolor='w', alpha=0.7)

X = df_merged[['Emissions (Trillions of Tonnes)']]
y = df_merged['Climate Vulnerability Score']
reg_model = LinearRegression()
reg_model.fit(X, y)
y_pred = reg_model.predict(X)
plt.plot(df_merged['Emissions (Trillions of Tonnes)'], y_pred, color='black', linewidth=2, label='Regression Line')

texts = []
for idx, row in df_merged.iterrows():
    x = row['Emissions (Trillions of Tonnes)']
    y_point = row['Climate Vulnerability Score']
    
    if row['Country Group'] == 'High Emission':
        x_text = x + 0.005
        ha = 'left'
    elif row['Country Group'] == 'High Vulnerability':
        x_text = x - 0.005
        ha = 'right'
    else:
        x_text = x
        ha = 'center'
    
    texts.append(plt.text(x_text, y_point, row['Country'], fontsize=10, ha=ha, va='center'))

adjust_text(texts, only_move={'points': 'y', 'text': 'xy'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.title('Total Aviation CO₂ Emissions vs. Climate Vulnerability Score (2019-2022)', fontsize=24)
plt.xlabel('Total Aviation CO₂ Emissions (Trillions of Tonnes)', fontsize=18)
plt.ylabel('Climate Vulnerability Score', fontsize=18)
plt.legend(title='Country Group', fontsize=14, title_fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

##Used Generative AI to help clean dataset