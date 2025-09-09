#!/usr/bin/env python3
"""
Restaurant Contact Matcher Script
Matches restaurant locations with contact information using both exact Google Places ID 
matching and fuzzy name matching. Errs on the side of including more contacts rather than missing any.
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

def normalize_restaurant_name(name):
    """
    Normalize restaurant names for better matching by removing common words,
    punctuation, and standardizing format.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ''
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove common restaurant-related words
    remove_words = [
        r'\b(the|a|an)\b',
        r'\b(restaurant|cafe|kitchen|bistro|grill|bar|tavern|pub|eatery|diner|pizzeria|brewery|winery|bakery|coffee|tea|house|shop|place|spot)\b',
        r'\b(inc|llc|ltd|corp|co|company|group|hospitality)\b',
    ]
    
    for pattern in remove_words:
        normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
    
    # Replace & with 'and'
    normalized = normalized.replace('&', 'and')
    
    # Remove all punctuation and special characters
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized

def calculate_similarity(str1, str2, method='combined'):
    """
    Calculate similarity between two strings using multiple methods.
    Returns a score between 0 and 1.
    """
    if pd.isna(str1) or pd.isna(str2) or not str1 or not str2:
        return 0
    
    norm1 = normalize_restaurant_name(str1)
    norm2 = normalize_restaurant_name(str2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return 1.0
    
    # Empty after normalization
    if not norm1 or not norm2:
        return 0
    
    scores = []
    
    # 1. Containment check (one string contains the other)
    if len(norm1) > 2 and len(norm2) > 2:
        if norm1 in norm2 or norm2 in norm1:
            min_len = min(len(norm1), len(norm2))
            max_len = max(len(norm1), len(norm2))
            containment_score = (min_len / max_len) * 0.9
            scores.append(containment_score)
    
    # 2. Word overlap
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if words1 and words2:
        # Filter out very short words
        words1 = {w for w in words1 if len(w) > 2}
        words2 = {w for w in words2 if len(w) > 2}
        
        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            # Jaccard similarity
            if union:
                word_score = len(intersection) / len(union)
                scores.append(word_score)
            
            # Also check for partial word matches for longer words
            partial_matches = 0
            for w1 in words1:
                for w2 in words2:
                    if len(w1) > 4 and len(w2) > 4:
                        if w1 in w2 or w2 in w1:
                            partial_matches += 0.5
            
            if len(words1) + len(words2) > 0:
                partial_score = partial_matches / max(len(words1), len(words2))
                if partial_score > 0:
                    scores.append(partial_score)
    
    # 3. Character sequence similarity (SequenceMatcher)
    sequence_score = SequenceMatcher(None, norm1, norm2).ratio()
    scores.append(sequence_score)
    
    # 4. Check if core words match (first significant word often contains the essence)
    words1_list = [w for w in norm1.split() if len(w) > 2]
    words2_list = [w for w in norm2.split() if len(w) > 2]
    
    if words1_list and words2_list:
        # Check if first significant words match
        if words1_list[0] == words2_list[0]:
            scores.append(0.7)
        
        # Check if any significant word matches exactly
        for w1 in words1_list[:3]:  # Check first 3 significant words
            if w1 in words2_list[:3]:
                scores.append(0.6)
                break
    
    # Return the maximum score from all methods
    return max(scores) if scores else 0

def match_restaurants_to_contacts(restaurants_df, contacts_df, threshold=0.35):
    """
    Match restaurants to contacts using both Google Places ID and fuzzy name matching.
    Uses a low threshold to be inclusive and capture all possible matches.
    """
    results = []
    
    print(f"Processing {len(restaurants_df)} restaurants against {len(contacts_df)} contacts...")
    print(f"Using similarity threshold: {threshold}")
    
    for idx, restaurant in restaurants_df.iterrows():
        restaurant_name = restaurant['Restaurant']
        google_place_id = restaurant['Google Places ID']
        
        matched_contacts = []
        
        # First, try exact Google Places ID matching
        if pd.notna(google_place_id) and google_place_id.strip():
            place_matches = contacts_df[
                (contacts_df['google_places_id'].notna()) & 
                (contacts_df['google_places_id'].str.strip() == google_place_id.strip())
            ]
            
            for _, contact in place_matches.iterrows():
                matched_contacts.append({
                    'contact': contact,
                    'match_type': 'Exact Google Place ID',
                    'match_score': 1.0
                })
        
        # Now do fuzzy name matching on Deal Name and Company Name
        for _, contact in contacts_df.iterrows():
            # Skip if already matched by Google Place ID
            if any(mc['contact']['Contact ID'] == contact['Contact ID'] and 
                   mc['match_type'] == 'Exact Google Place ID' for mc in matched_contacts):
                continue
            
            best_score = 0
            best_match_field = ''
            
            # Check Deal Name
            if pd.notna(contact['Deal Name']):
                deal_score = calculate_similarity(restaurant_name, contact['Deal Name'])
                if deal_score > best_score:
                    best_score = deal_score
                    best_match_field = 'Deal Name'
            
            # Check Company Name
            if pd.notna(contact['Company name']):
                company_score = calculate_similarity(restaurant_name, contact['Company name'])
                if company_score > best_score:
                    best_score = company_score
                    best_match_field = 'Company Name'
            
            # Add if score meets threshold
            if best_score >= threshold:
                matched_contacts.append({
                    'contact': contact,
                    'match_type': f'Fuzzy {best_match_field}',
                    'match_score': best_score
                })
        
        # Sort matches by score (highest first)
        matched_contacts.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Create output rows
        if not matched_contacts:
            # No matches found - add a row indicating this
            results.append({
                'Restaurant': restaurant_name,
                'City': restaurant.get('City', ''),
                'Google Places ID': google_place_id if pd.notna(google_place_id) else '',
                'TPV 90 Days': restaurant.get('TPV 90 Days', ''),
                'Red Flags': restaurant.get('Red Flags', ''),
                'Cohort': restaurant.get('Cohort', ''),
                'Match Status': 'No Matches Found',
                'Match Type': '',
                'Match Score': 0,
                'Contact Email': '',
                'Contact First Name': '',
                'Contact Last Name': '',
                'Contact Company': '',
                'Contact Deal Name': '',
                'Contact Google Places ID': '',
                'Company ID': '',
                'Contact ID': '',
                'Deal ID': '',
                'In Other List?': '',
                'Macro Geo': ''
            })
        else:
            # Add all matched contacts
            for match in matched_contacts:
                contact = match['contact']
                results.append({
                    'Restaurant': restaurant_name,
                    'City': restaurant.get('City', ''),
                    'Google Places ID': google_place_id if pd.notna(google_place_id) else '',
                    'TPV 90 Days': restaurant.get('TPV 90 Days', ''),
                    'Red Flags': restaurant.get('Red Flags', ''),
                    'Cohort': restaurant.get('Cohort', ''),
                    'Match Status': 'Matched',
                    'Match Type': match['match_type'],
                    'Match Score': f"{match['match_score']:.3f}",
                    'Contact Email': contact.get('Email', ''),
                    'Contact First Name': contact.get('first name', ''),
                    'Contact Last Name': contact.get('Last Name', ''),
                    'Contact Company': contact.get('Company name', ''),
                    'Contact Deal Name': contact.get('Deal Name', ''),
                    'Contact Google Places ID': contact.get('google_places_id', ''),
                    'Company ID': contact.get('Company ID', ''),
                    'Contact ID': contact.get('Contact ID', ''),
                    'Deal ID': contact.get('Deal ID', ''),
                    'In Other List?': contact.get('In Other List?', ''),
                    'Macro Geo': contact.get('Macro Geo (NYC, SF, CHS, DC, LA, NASH, DEN)', '')
                })
        
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(restaurants_df)} restaurants...")
    
    return pd.DataFrame(results)

def main():
    """
    Main function to run the restaurant contact matching process.
    """
    print("=" * 60)
    print("Restaurant Contact Matcher")
    print("=" * 60)
    
    # File paths - update these to match your file locations
    RESTAURANTS_FILE = "restaurantinfo.csv"
    CONTACTS_FILE = "contactinfo.csv"
    OUTPUT_FILE = "restaurant_contact_matches.csv"
    
    # You can also use command line arguments if preferred
    import sys
    if len(sys.argv) == 3:
        RESTAURANTS_FILE = sys.argv[1]
        CONTACTS_FILE = sys.argv[2]
    elif len(sys.argv) == 4:
        RESTAURANTS_FILE = sys.argv[1]
        CONTACTS_FILE = sys.argv[2]
        OUTPUT_FILE = sys.argv[3]
    
    try:
        # Load the data
        print(f"\nLoading restaurant data from: {RESTAURANTS_FILE}")
        restaurants_df = pd.read_csv(RESTAURANTS_FILE)
        print(f"Loaded {len(restaurants_df)} restaurants")
        
        print(f"\nLoading contact data from: {CONTACTS_FILE}")
        contacts_df = pd.read_csv(CONTACTS_FILE)
        print(f"Loaded {len(contacts_df)} contacts")
        
        # Perform matching
        print("\n" + "=" * 60)
        print("Starting matching process...")
        print("=" * 60)
        
        results_df = match_restaurants_to_contacts(
            restaurants_df, 
            contacts_df,
            threshold=0.35  # Low threshold to be inclusive
        )
        
        # Save results
        print(f"\nSaving results to: {OUTPUT_FILE}")
        results_df.to_csv(OUTPUT_FILE, index=False)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("MATCHING COMPLETE - SUMMARY")
        print("=" * 60)
        
        total_restaurants = len(restaurants_df)
        matched_restaurants = results_df[results_df['Match Status'] == 'Matched']['Restaurant'].nunique()
        no_match_restaurants = results_df[results_df['Match Status'] == 'No Matches Found']['Restaurant'].nunique()
        total_matches = len(results_df[results_df['Match Status'] == 'Matched'])
        
        print(f"Total restaurants processed: {total_restaurants}")
        print(f"Restaurants with matches: {matched_restaurants} ({matched_restaurants/total_restaurants*100:.1f}%)")
        print(f"Restaurants with no matches: {no_match_restaurants}")
        print(f"Total matched contacts: {total_matches}")
        print(f"Average contacts per restaurant: {total_matches/total_restaurants:.1f}")
        
        # Show match type distribution
        print("\nMatch Type Distribution:")
        match_types = results_df[results_df['Match Status'] == 'Matched']['Match Type'].value_counts()
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count}")
        
        # Show score distribution for fuzzy matches
        fuzzy_matches = results_df[results_df['Match Type'].str.contains('Fuzzy', na=False)]
        if not fuzzy_matches.empty:
            fuzzy_matches['Score'] = fuzzy_matches['Match Score'].astype(float)
            print("\nFuzzy Match Score Distribution:")
            print(f"  Minimum: {fuzzy_matches['Score'].min():.3f}")
            print(f"  Maximum: {fuzzy_matches['Score'].max():.3f}")
            print(f"  Mean: {fuzzy_matches['Score'].mean():.3f}")
            print(f"  Median: {fuzzy_matches['Score'].median():.3f}")
        
        print(f"\n✅ Results saved to: {OUTPUT_FILE}")
        print(f"Total output rows: {len(results_df)}")
        
        # Optional: Save high-confidence matches separately
        high_confidence = results_df[
            (results_df['Match Type'] == 'Exact Google Place ID') | 
            (results_df['Match Score'].astype(float) >= 0.6)
        ]
        
        if not high_confidence.empty:
            high_conf_file = OUTPUT_FILE.replace('.csv', '_high_confidence.csv')
            high_confidence.to_csv(high_conf_file, index=False)
            print(f"\n📊 High confidence matches (>60% or exact) saved to: {high_conf_file}")
            print(f"   Contains {len(high_confidence)} rows")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file - {e}")
        print("\nPlease ensure the following files are in the current directory:")
        print(f"  1. {RESTAURANTS_FILE}")
        print(f"  2. {CONTACTS_FILE}")
        print("\nOr run the script with file paths as arguments:")
        print("  python script.py 'restaurants.csv' 'contacts.csv' 'output.csv'")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()