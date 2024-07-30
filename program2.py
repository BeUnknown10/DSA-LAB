import pandas as pd
import numpy as np 
import re

# Import the data into a DataFrame
books_df = pd.read_csv('BL-Flickr-Images-Book.csv')

# Display the first few rows of the DataFrame
print("Original DataFrame:") 
print(books_df.head())

# Find and drop the columns which are irrelevant for the book information
columns_to_drop = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner','Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']
books_df.drop(columns=columns_to_drop, inplace=True) 

# Change the Index of the DataFrame
books_df.set_index('Identifier', inplace=True)

# Tidy up fields in the data such as date of publication with the help of simple regular expression 
def clean_date(date): 
    if isinstance(date, str):
        match = re.search(r'\d{4}', date)
        if match:
            return match.group()
    return np.nan
books_df['Date of Publication'] = books_df['Date of Publication'].apply(clean_date) 

# Combine str methods with NumPy to clean columns
books_df['Place of Publication'] = np.where( 
    books_df['Place of Publication'].str.contains('London'), 
      'London',
    np.where(
        books_df['Place of Publication'].str.contains('Oxford'),
        'Oxford',
        books_df['Place of Publication'].replace(
            r'^\s*$', 'Unknown', regex=True        )
    )
)
# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(books_df.head())

