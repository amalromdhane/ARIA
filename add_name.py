"""
Temporary: Manually add name to pending visitor
"""

import pickle
import sys

# Load database
with open('visitor_database.pkl', 'rb') as f:
    visitors = pickle.load(f)

# Find newest visitor with "New Visitor" name
for vid, data in sorted(visitors.items(), reverse=True):
    if data['name'] == "New Visitor":
        print(f"Found pending visitor: {vid}")
        
        name = input("Enter their name: ")
        
        visitors[vid]['name'] = name
        
        # Save
        with open('visitor_database.pkl', 'wb') as f:
            pickle.dump(visitors, f)
        
        print(f"✅ Updated {vid} → {name}")
        break
else:
    print("No pending visitors found")
