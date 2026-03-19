#!/usr/bin/env python3
"""
Manual name setter - Use when voice recognition doesn't work
"""

import pickle
import sys

def set_name(visitor_id, name):
    db_file = 'visitor_database.pkl'
    
    if not os.path.exists(db_file):
        print("❌ No database found!")
        return
    
    with open(db_file, 'rb') as f:
        visitors = pickle.load(f)
    
    if visitor_id in visitors:
        old_name = visitors[visitor_id].get('name', 'Unknown')
        visitors[visitor_id]['name'] = name
        
        with open(db_file, 'wb') as f:
            pickle.dump(visitors, f)
        
        print(f"✅ Changed: {visitor_id}")
        print(f"   From: {old_name}")
        print(f"   To:   {name}")
    else:
        print(f"❌ {visitor_id} not found!")
        print(f"Available visitors:")
        for vid in visitors.keys():
            vname = visitors[vid].get('name', 'Unknown')
            print(f"   {vid}: {vname}")

if __name__ == "__main__":
    import os
    
    if len(sys.argv) != 3:
        print("Usage: python3 set_name.py <visitor_id> 'Name'")
        print("Example: python3 set_name.py Visitor_001 'Malak'")
        print("\nCurrent visitors:")
        
        if os.path.exists('visitor_database.pkl'):
            with open('visitor_database.pkl', 'rb') as f:
                visitors = pickle.load(f)
            for vid in visitors.keys():
                vname = visitors[vid].get('name', 'Unknown')
                print(f"   {vid}: {vname}")
        else:
            print("   (No database yet)")
        sys.exit(1)
    
    set_name(sys.argv[1], sys.argv[2])

