from datetime import datetime

from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['test_db']
collection = db['test_collection']

# Insert a test document
test_data = {'name': 'Test Document', 'date': datetime.now()}
result = collection.insert_one(test_data)
print(f"Inserted document with ID: {result.inserted_id}")

for doc in collection.find():
    print(doc)

