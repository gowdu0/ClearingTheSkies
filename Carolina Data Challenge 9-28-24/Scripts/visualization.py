import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

df = pd.read_excel('Data/emissions.xlsx')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.fillna(0, inplace=True)

emissions_columns = df.columns.drop(['Country', 'Climate Vulnerability Score'])
emissions_long = df.melt(
    id_vars=['Country', 'Climate Vulnerability Score'],
    value_vars=emissions_columns,
    var_name='Date',
    value_name='Emissions'
)
emissions_long['Date'] = pd.to_datetime(emissions_long['Date'], format='%Y-%b')
emissions_long['Emissions'] = pd.to_numeric(emissions_long['Emissions'], errors='coerce').fillna(0)
emissions_long['Emissions (Trillions of Tonnes)'] = emissions_long['Emissions'] / 1_000_000_000_000

total_emissions = emissions_long.groupby('Country')['Emissions (Trillions of Tonnes)'].sum().reset_index()

df_merged = pd.merge(
    total_emissions,
    df[['Country', 'Climate Vulnerability Score']].drop_duplicates(),
    on='Country'
)

country_name_mapping = {
    'United States': 'United States of America',
    'South Korea': 'Republic of Korea',
    'Czech Republic': 'Czechia',
    'North Korea': "Democratic People's Republic of Korea",
    'Syria': 'Syrian Arab Republic',
    'Iran': 'Iran (Islamic Republic of)',
    'Bolivia': 'Bolivia (Plurinational State of)',
    'Venezuela': 'Venezuela (Bolivarian Republic of)',
}
df_merged['Country'] = df_merged['Country'].replace(country_name_mapping)

world = gpd.read_file('Data/ne_110m_admin_0_countries.shp')
world = world.merge(df_merged, left_on='NAME', right_on='Country', how='left')

emissions_threshold = world['Emissions (Trillions of Tonnes)'].quantile(0.25)
vulnerability_threshold = world['Climate Vulnerability Score'].quantile(0.75)
world['Impact'] = 'Other'
world.loc[
    (world['Emissions (Trillions of Tonnes)'] <= emissions_threshold) &
    (world['Climate Vulnerability Score'] >= vulnerability_threshold),
    'Impact'
] = 'Disproportionately Impacted'

fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

cmap = plt.cm.Reds
emission_vmin = 0.1
emission_vmax = 0.6
norm = Normalize(vmin=emission_vmin, vmax=emission_vmax)

for idx, row in world.iterrows():
    geom = row['geometry']
    if pd.isnull(row['Emissions (Trillions of Tonnes)']):
        facecolor = 'lightgrey'
    else:
        emission = row['Emissions (Trillions of Tonnes)']
        facecolor = cmap(norm(emission))
    if row['Impact'] == 'Disproportionately Impacted':
        facecolor = 'green'
    if row['NAME'] == 'United States of America' and not pd.isnull(row['Emissions (Trillions of Tonnes)']):
        facecolor = cmap(norm(emission_vmax))
    if geom is None:
        continue
    if geom.type == 'Polygon':
        polygons = [geom]
    elif geom.type == 'MultiPolygon':
        polygons = geom.geoms
    else:
        continue
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        poly3d = [[(x_coord, y_coord, 0) for x_coord, y_coord in poly.exterior.coords]]
        collection = Poly3DCollection(poly3d, facecolors=facecolor, edgecolors='black', linewidths=0.5, alpha=0.6)
        ax.add_collection3d(collection)

world['centroid'] = world['geometry'].centroid
world['lon'] = world['centroid'].x
world['lat'] = world['centroid'].y

scale_factor = 10

for idx, row in world.dropna(subset=['Emissions (Trillions of Tonnes)', 'Climate Vulnerability Score']).iterrows():
    x = row['lon']
    y = row['lat']
    z_start = 0
    z_end = row['Climate Vulnerability Score'] * scale_factor
    ax.plot([x, x], [y, y], [z_start, z_end], color='blue', linewidth=2, alpha=0.7)

for idx, row in world.dropna(subset=['Emissions (Trillions of Tonnes)', 'Climate Vulnerability Score']).iterrows():
    if row['Impact'] == 'Disproportionately Impacted':
        x = row['lon']
        y = row['lat']
        z = row['Climate Vulnerability Score'] * scale_factor
        country_name = row['NAME']
        ax.text(
            x,
            y,
            z,
            country_name,
            fontsize=10,
            ha='center',
            va='bottom',
            color='black',
            weight='bold'
        )

ax.set_xlabel('Longitude', fontsize=14, labelpad=15)
ax.set_ylabel('Latitude', fontsize=14, labelpad=15)
ax.set_zlabel('Climate Vulnerability Score', fontsize=14, labelpad=15)
plt.title('Global Aviation CO₂ Emissions and Climate Vulnerability (2019-2022)', fontsize=24, pad=30)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
cbar.set_label('Total Aviation CO₂ Emissions (Trillions of Tonnes)', fontsize=12)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
cbar.ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

extrusion_patch = mpatches.Patch(color='blue', label='Climate Vulnerability Score')
impact_patch = mpatches.Patch(color='green', label='Disproportionately Impacted')
plt.legend(handles=[extrusion_patch, impact_patch], loc='upper left', fontsize=12)

ax.view_init(elev=30, azim=-60)

fig.patch.set_facecolor('white')
ax.set_facecolor('lightblue')

ax.set_zlim(0, scale_factor * 1.0)
z_ticks = np.linspace(0, scale_factor, 6)
z_ticklabels = [f'{tick/scale_factor:.1f}' for tick in z_ticks]
ax.set_zticks(z_ticks)
ax.set_zticklabels(z_ticklabels)

plt.savefig('3D_Aviation_CO2_Emissions_Vulnerability_Presentation.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
