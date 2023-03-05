import pandas as pd
import numpy as np
df = pd.read_table("tsv.tsv")
df.info()

# Convert object columns to float
cols_to_convert = ["AlogP", "Polar Surface Area", "HBA", "HBD", "#RO5 Violations", "#Rotatable Bonds", 
                   "QED Weighted", "CX Acidic pKa", "CX Basic pKa", "CX LogP", "CX LogD", "Aromatic Rings", 
                   "Heavy Atoms", "HBA (Lipinski)", "HBD (Lipinski)", "#RO5 Violations (Lipinski)", 
                   "Molecular Weight (Monoisotopic)", "Np Likeness Score"]
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

# Drop irrelevant columns
cols_to_drop = ["Max Phase", "Inorganic Flag", "Structure Type", "Molecular Species"]
df = df.drop(columns=cols_to_drop)

# Drop rows with missing values
df = df.dropna()

# Rename the Chemical ID column
df = df.rename(columns={"ChEMBL ID": "Chemical ID"})

# Set the Chemical ID column as index
df = df.set_index("Chemical ID")

# Drop duplicates
df = df.drop_duplicates()