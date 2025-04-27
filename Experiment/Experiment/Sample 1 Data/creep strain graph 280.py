# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:24:31 2025

@author: camer
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load CSV file
df = pd.read_csv('Sample1_280MPa_550C_200h_code.csv')

# Extract relevant columns
time = df['Test time']
strain = df['Standard travel'] / 100  # Convert to strain
stress = df['Standard force']

# Identify the initial elastic and plastic regions (adjust range as needed)
elastic_region = df.iloc[29:264]
plastic_region = df.iloc[487:907]

# Perform linear regression to estimate Elastic Modulus (E)
slope, intercept, r_value, _, _ = linregress(elastic_region['Standard travel'] / 100, elastic_region['Standard force'])
E = slope  # Elastic modulus in MPa

slope, intercept, r_value, _, _ = linregress(plastic_region['Standard travel'] / 100, plastic_region['Standard force'])
H = slope  # Plastic modulus in MPa

# Print estimated Young’s and Hardening modulus
print(f"Estimated Elastic Modulus (E): {E:.2f} MPa")
print(f"Estimated Plastic Modulus (H): {H:.2f} MPa")

yield_stress = 137.8

# Compute elastic and plastic strain
elastic_strain = stress / E  # Elastic strain
plastic_strain = 0.2 * (stress - yield_stress) / H  # Plastic strain including strain hardening exponent (0.2)

# Compute creep strain
creep_strain = strain - elastic_strain - plastic_strain

# Add creep strain to dataframe
df['Creep Strain'] = creep_strain

# Filter to keep only positive creep strain values
df_positive_creep = df[df['Creep Strain'] >= 0].copy()

# Filter to only include data where Standard force is > 278 MPa and time ≤ 200 hours
df_positive_creep = df_positive_creep[df_positive_creep['Standard force'] > 278]

df_positive_creep = df_positive_creep[df_positive_creep['Test time'] <= 201]

# Apply a moving average filter to smooth the creep strain data
window_size = 10  # Adjust for more/less smoothing
df_positive_creep['Smoothed Creep Strain'] = df_positive_creep['Creep Strain'].rolling(window=window_size, center=True).mean()

# Normalize creep strain so that the minimum value in the range starts at zero
min_creep_strain = df_positive_creep['Smoothed Creep Strain'].min()
df_positive_creep['Normalized Creep Strain'] = df_positive_creep['Smoothed Creep Strain'] - min_creep_strain
df_positive_creep['Normalized Original Creep Strain'] = df_positive_creep['Creep Strain'] - min_creep_strain

# Find the maximum creep strain and corresponding time
max_creep_strain = df_positive_creep['Normalized Creep Strain'].max()
max_creep_time = df_positive_creep[df_positive_creep['Normalized Creep Strain'] == max_creep_strain]['Test time'].values[0]
print(f"Maximum Normalized Creep Strain: {max_creep_strain:.6f} at Time: {max_creep_time} hours")

# Define the time range
start_time = 3.65
end_time = 200

# Find indices for the time range
start_index = np.searchsorted(time, start_time)
end_index = np.searchsorted(time, end_time)

# Extract the strain values for the time range
strain_range = strain.iloc[start_index:end_index + 1]

# Find the maximum strain in the time range
max_strain = strain_range.max()

# Calculate the strain at the start time
strain_start = strain.iloc[start_index]

# Calculate total strain increase
strain_increase_max = max_strain - strain_start
print(f"Maximum strain increase between {start_time} hours and {end_time} hours: {strain_increase_max:.6f}")

# Plot the normalized creep strain graph
plt.figure(figsize=(9, 5))
plt.plot(df_positive_creep['Test time'], df_positive_creep['Normalized Creep Strain'], linestyle='-', label='Smoothed Creep Strain')
plt.plot(df_positive_creep['Test time'], df_positive_creep['Normalized Original Creep Strain'], linestyle='-', alpha=0.3, label='Original Creep Strain', color='gray')

# Labels and title
plt.xlabel('Time (h)')
plt.ylabel('Creep Strain)')
plt.grid(True)
plt.legend()
plt.show()
