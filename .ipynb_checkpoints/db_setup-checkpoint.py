# db_setup.py
import pandas as pd
from sqlalchemy import create_engine

# Load dataset
df = pd.read_csv('amazon_reviews.csv', low_memory=False)

# Create SQLite engine (creates file 'reviews.db')
engine = create_engine('sqlite:///reviews.db')

# Save DataFrame to SQL table 'reviews'
df.to_sql('reviews', con=engine, if_exists='replace', index=False)

print("Database setup complete, reviews saved to 'reviews.db'")
