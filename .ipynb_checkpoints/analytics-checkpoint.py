# analytics.py
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///reviews.db')

# Example 1: Count complaints (rating < 3) by brand
query1 = """
SELECT brand, COUNT(*) AS complaint_count
FROM reviews
WHERE "reviews.rating" < 3
GROUP BY brand
ORDER BY complaint_count DESC
LIMIT 10
"""
df_complaints = pd.read_sql(query1, engine)
print("Top 10 brands by complaints:")
print(df_complaints)

# Example 2: Top 10 most complained products
query2 = """
SELECT name, COUNT(*) AS complaints
FROM reviews
WHERE "reviews.rating" < 3
GROUP BY name
ORDER BY complaints DESC
LIMIT 10
"""
df_products = pd.read_sql(query2, engine)
print("\nTop 10 products by complaints:")
print(df_products)

# Example 3: Complaints over time (monthly count)
query3 = """
SELECT substr("reviews.date", 1, 7) AS month, COUNT(*) AS complaints
FROM reviews
WHERE "reviews.rating" < 3
GROUP BY month
ORDER BY month
"""
df_time = pd.read_sql(query3, engine)
print("\nComplaints over time (monthly):")
print(df_time)
