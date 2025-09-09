#!/usr/bin/env python3
"""
Restaurant Contact Matcher Script with Claude Verification
Matches restaurant locations with contact information using both exact Google Places ID 
matching and fuzzy name matching, then verifies matches using Claude Sonnet 4.
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import warnings
import json
from anthropic import Anthropic
import time
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

# Configuration constants
INCLUSIVE_THRESHOLD = 0.35  # Threshold for fuzzy matching - lower values are more inclusive
BATCH_SIZE = 20  # Number of pairings to send to Claude at once
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"  # Claude Sonnet 4 model

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

def verify_matches_with_claude(results_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Use Claude Sonnet 4 to verify that matched restaurant-contact pairs 
    actually refer to the same establishment.
    """
    print("\n" + "=" * 60)
    print("CLAUDE VERIFICATION STEP")
    print("=" * 60)
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Get only matched rows (skip "No Matches Found")
    matched_df = results_df[results_df['Match Status'] == 'Matched'].copy()
    
    if matched_df.empty:
        print("No matches to verify.")
        return results_df
    
    # Skip verification for exact Google Place ID matches (these are definitionally correct)
    exact_matches = matched_df[matched_df['Match Type'] == 'Exact Google Place ID']
    fuzzy_matches = matched_df[matched_df['Match Type'] != 'Exact Google Place ID']
    
    print(f"Skipping verification for {len(exact_matches)} exact Google Place ID matches")
    print(f"Verifying {len(fuzzy_matches)} fuzzy matches")
    
    if fuzzy_matches.empty:
        print("No fuzzy matches to verify.")
        return results_df
    
    # Get unique pairings to verify
    unique_pairings = fuzzy_matches[['Restaurant', 'City', 'Contact Deal Name', 'Contact Company']].drop_duplicates()
    
    # Filter out rows where Contact Deal Name is empty
    unique_pairings = unique_pairings[unique_pairings['Contact Deal Name'].notna() & (unique_pairings['Contact Deal Name'] != '')]
    
    print(f"Found {len(unique_pairings)} unique restaurant-contact pairings to verify")
    
    # Process in batches
    verified_pairings = {}
    
    for i in range(0, len(unique_pairings), BATCH_SIZE):
        batch = unique_pairings.iloc[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(unique_pairings) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\nProcessing batch {batch_num}/{total_batches}...")
        
        # Prepare batch for Claude
        batch_items = []
        for _, row in batch.iterrows():
            item = {
                "restaurant_name": row['Restaurant'],
                "restaurant_city": row['City'] if pd.notna(row['City']) else "Unknown",
                "contact_deal_name": row['Contact Deal Name'],
                "contact_company": row['Contact Company'] if pd.notna(row['Contact Company']) else ""
            }
            batch_items.append(item)
        
        # Create prompt for Claude
        prompt = f"""You are a restaurant verification expert. I need you to determine if the following restaurant names and contact deal names actually refer to the SAME physical establishment or business.

For each pairing, consider:
1. Are these clearly the same business (even if names differ slightly)?
2. Could one be a parent company/franchise and the other a specific location?
3. Are there any indicators these might be DIFFERENT businesses (e.g., completely different cuisine types, one is clearly a chain and the other isn't)?
4. Consider the city location if provided

Please respond with a JSON array where each item has:
- "index": the index number (0-based)
- "same_place": true/false (true if they refer to the same establishment)
- "confidence": "high", "medium", or "low"
- "reason": brief explanation (max 50 words)

Here are the pairings to verify:

{json.dumps(batch_items, indent=2)}

Return ONLY the JSON array, no other text."""

        try:
            # Call Claude API
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse Claude's response
            response_text = response.content[0].text.strip()
            
            # Try to extract JSON if it's wrapped in markdown
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            verification_results = json.loads(response_text)
            
            # Store results
            for j, result in enumerate(verification_results):
                if j < len(batch):
                    row = batch.iloc[j]
                    key = (row['Restaurant'], row['Contact Deal Name'])
                    verified_pairings[key] = result['same_place']
                    
                    if not result['same_place']:
                        print(f"  ❌ Rejected: {row['Restaurant']} ≠ {row['Contact Deal Name']}")
                        print(f"     Reason: {result['reason']}")
            
            # Rate limiting
            time.sleep(0.5)  # Be respectful of API limits
            
        except Exception as e:
            print(f"  ⚠️ Error processing batch {batch_num}: {e}")
            # On error, be conservative and keep the matches
            for _, row in batch.iterrows():
                key = (row['Restaurant'], row['Contact Deal Name'])
                verified_pairings[key] = True
    
    # Apply verification results
    print("\n" + "-" * 60)
    print("Applying verification results...")
    
    rows_to_keep = []
    removed_count = 0
    
    for _, row in results_df.iterrows():
        # Always keep non-matched rows and exact Google Place ID matches
        if row['Match Status'] != 'Matched' or row['Match Type'] == 'Exact Google Place ID':
            rows_to_keep.append(True)
        else:
            # Check if this pairing was verified
            key = (row['Restaurant'], row['Contact Deal Name'])
            if key in verified_pairings:
                keep = verified_pairings[key]
                rows_to_keep.append(keep)
                if not keep:
                    removed_count += 1
            else:
                # If not in verification (e.g., empty deal name), keep it
                rows_to_keep.append(True)
    
    # Filter the dataframe
    verified_df = results_df[rows_to_keep].copy()
    
    print(f"Removed {removed_count} rows that failed verification")
    print(f"Final dataset contains {len(verified_df)} rows")
    
    return verified_df

def main():
    """
    Main function to run the restaurant contact matching process with Claude verification.
    """
    print("=" * 60)
    print("Restaurant Contact Matcher with Claude Verification")
    print("=" * 60)
    
    # File paths - update these to match your file locations
    RESTAURANTS_FILE = "restaurantinfo.csv"
    CONTACTS_FILE = "contactinfo.csv"
    OUTPUT_FILE = "restaurant_contact_matches.csv"
    
    # You can also use command line arguments if preferred
    import sys
    import os
    
    if len(sys.argv) >= 3:
        RESTAURANTS_FILE = sys.argv[1]
        CONTACTS_FILE = sys.argv[2]
    if len(sys.argv) >= 4:
        OUTPUT_FILE = sys.argv[3]
    
    # Get API key from environment variable or prompt
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not found in environment variables.")
        api_key = input("Please enter your Anthropic API key: ").strip()
        if not api_key:
            print("❌ API key is required for Claude verification. Exiting.")
            return
    
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
        print("STEP 1: Initial Matching Process")
        print("=" * 60)
        
        results_df = match_restaurants_to_contacts(
            restaurants_df, 
            contacts_df,
            threshold=INCLUSIVE_THRESHOLD
        )
        
        # Save pre-verification results
        pre_verify_file = OUTPUT_FILE.replace('.csv', '_pre_verification.csv')
        results_df.to_csv(pre_verify_file, index=False)
        print(f"\nPre-verification results saved to: {pre_verify_file}")
        
        # Verify matches with Claude
        print("\n" + "=" * 60)
        print("STEP 2: Claude Verification")
        print("=" * 60)
        
        verified_df = verify_matches_with_claude(results_df, api_key)
        
        # Save final results
        print(f"\nSaving verified results to: {OUTPUT_FILE}")
        verified_df.to_csv(OUTPUT_FILE, index=False)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("MATCHING COMPLETE - FINAL SUMMARY")
        print("=" * 60)
        
        total_restaurants = len(restaurants_df)
        matched_restaurants = verified_df[verified_df['Match Status'] == 'Matched']['Restaurant'].nunique()
        no_match_restaurants = verified_df[verified_df['Match Status'] == 'No Matches Found']['Restaurant'].nunique()
        total_matches = len(verified_df[verified_df['Match Status'] == 'Matched'])
        
        print(f"Total restaurants processed: {total_restaurants}")
        print(f"Restaurants with verified matches: {matched_restaurants} ({matched_restaurants/total_restaurants*100:.1f}%)")
        print(f"Restaurants with no matches: {no_match_restaurants}")
        print(f"Total verified matched contacts: {total_matches}")
        print(f"Average contacts per restaurant: {total_matches/total_restaurants:.1f}")
        
        # Show match type distribution
        print("\nVerified Match Type Distribution:")
        match_types = verified_df[verified_df['Match Status'] == 'Matched']['Match Type'].value_counts()
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count}")
        
        # Show score distribution for fuzzy matches
        fuzzy_matches = verified_df[verified_df['Match Type'].str.contains('Fuzzy', na=False)]
        if not fuzzy_matches.empty:
            fuzzy_matches['Score'] = fuzzy_matches['Match Score'].astype(float)
            print("\nVerified Fuzzy Match Score Distribution:")
            print(f"  Minimum: {fuzzy_matches['Score'].min():.3f}")
            print(f"  Maximum: {fuzzy_matches['Score'].max():.3f}")
            print(f"  Mean: {fuzzy_matches['Score'].mean():.3f}")
            print(f"  Median: {fuzzy_matches['Score'].median():.3f}")
        
        print(f"\n✅ Verified results saved to: {OUTPUT_FILE}")
        print(f"Total output rows: {len(verified_df)}")
        
        # Compare pre and post verification
        print("\n📊 Verification Impact:")
        print(f"  Rows before verification: {len(results_df)}")
        print(f"  Rows after verification: {len(verified_df)}")
        print(f"  Rows removed: {len(results_df) - len(verified_df)}")
        
        # Optional: Save high-confidence matches separately
        high_confidence = verified_df[
            (verified_df['Match Type'] == 'Exact Google Place ID') | 
            (verified_df['Match Score'].astype(str) != '0') & (verified_df['Match Score'].astype(float) >= 0.6)
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