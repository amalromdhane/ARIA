import pickle
import os

db_file = 'visitor_database.pkl'

if os.path.exists(db_file):
    with open(db_file, 'rb') as f:
        visitors = pickle.load(f)
    
    print(f"Total visitors: {len(visitors)}")
    print("="*60)
    
    for visitor_id, data in visitors.items():
        print(f"\nID: {visitor_id}")
        print(f"  Name: {data.get('name', 'Unknown')}")
        print(f"  Visits: {data.get('visits', 0)}")
        print(f"  First seen: {data.get('first_seen', 'Unknown')}")
        print(f"  Last seen: {data.get('last_seen', 'Unknown')}")
        print("-"*40)
else:
    print("No database found!")
