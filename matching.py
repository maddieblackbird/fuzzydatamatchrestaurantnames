#!/usr/bin/env python3
"""
Improved Restaurant Contact Matcher with Smarter Fuzzy Matching
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import warnings
from typing import Tuple, List

warnings.filterwarnings('ignore')

def normalize_restaurant_name(name):
    """
    Normalize restaurant names for better matching.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ''
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove common restaurant-related words that don't help with identity
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

def extract_core_name(name):
    """
    Extract the core identifying part of a restaurant name.
    This helps identify the actual business identity.
    """
    if pd.isna(name) or not isinstance(name, str):
        return ''
    
    # Normalize first
    normalized = normalize_restaurant_name(name)
    
    # Get the first 2-3 significant words (often the actual restaurant name)
    words = [w for w in normalized.split() if len(w) > 2]
    
    if not words:
        return normalized
    
    # Return first 2 significant words as the "core"
    return ' '.join(words[:2])

def calculate_similarity_improved(name1, name2, city1=None, city2=None):
    """
    Improved similarity calculation with better logic.
    Returns (score, match_type, confidence)
    """
    if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
        return 0, "No Match", "none"
    
    norm1 = normalize_restaurant_name(name1)
    norm2 = normalize_restaurant_name(name2)
    
    # Extract core names
    core1 = extract_core_name(name1)
    core2 = extract_core_name(name2)
    
    # Check for exact match after normalization
    if norm1 == norm2:
        return 1.0, "Exact Name Match", "high"
    
    # Check if core names match exactly
    if core1 and core2 and core1 == core2:
        # If cities match too, very high confidence
        if city1 and city2 and city1.lower() == city2.lower():
            return 0.95, "Core Name + City Match", "high"
        return 0.85, "Core Name Match", "high"
    
    # Check for substring containment (one name contains the other)
    if len(norm1) > 4 and len(norm2) > 4:
        if norm1 in norm2 or norm2 in norm1:
            min_len = min(len(norm1), len(norm2))
            max_len = max(len(norm1), len(norm2))
            containment_score = min_len / max_len
            if containment_score > 0.7:
                return containment_score, "Name Contains Other", "medium"
    
    # Word-based matching
    words1 = set(w for w in norm1.split() if len(w) > 2)
    words2 = set(w for w in norm2.split() if len(w) > 2)
    
    if words1 and words2:
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Need at least one meaningful word in common
        if len(intersection) > 0:
            jaccard = len(intersection) / len(union)
            
            # Check if the matching words are significant (the main identifier)
            first_word_match = False
            if norm1.split() and norm2.split():
                first_words_1 = [w for w in norm1.split() if len(w) > 2][:1]
                first_words_2 = [w for w in norm2.split() if len(w) > 2][:1]
                if first_words_1 and first_words_2 and first_words_1[0] == first_words_2[0]:
                    first_word_match = True
            
            if first_word_match and jaccard > 0.5:
                return jaccard, "First Word + Partial Match", "medium"
            elif jaccard > 0.7:
                return jaccard, "High Word Overlap", "medium"
            elif jaccard > 0.5:
                return jaccard * 0.8, "Moderate Word Overlap", "low"
    
    # Character sequence similarity as last resort
    sequence_score = SequenceMatcher(None, norm1, norm2).ratio()
    
    if sequence_score > 0.8:
        return sequence_score, "High Character Similarity", "low"
    elif sequence_score > 0.65:
        return sequence_score * 0.7, "Moderate Character Similarity", "low"
    
    return 0, "No Match", "none"

def match_restaurants_with_validation(restaurants_df, contacts_df, 
                                     strict_threshold=0.65,
                                     verify_threshold=0.8):
    """
    Match restaurants to contacts with improved logic.
    
    Args:
        strict_threshold: Minimum score to consider a match (default 0.65)
        verify_threshold: Score below which matches need verification (default 0.8)
    """
    results = []
    verification_needed = []
    
    print(f"Processing {len(restaurants_df)} restaurants against {len(contacts_df)} contacts...")
    print(f"Match threshold: {strict_threshold}")
    print(f"Verification threshold: {verify_threshold}")
    
    for idx, restaurant in restaurants_df.iterrows():
        restaurant_name = restaurant['Restaurant']
        restaurant_city = restaurant.get('City', '')
        google_place_id = restaurant['Google Places ID']
        
        matched_contacts = []
        
        # First, try exact Google Places ID matching (always reliable)
        if pd.notna(google_place_id) and google_place_id.strip():
            place_matches = contacts_df[
                (contacts_df['google_places_id'].notna()) & 
                (contacts_df['google_places_id'].str.strip() == google_place_id.strip())
            ]
            
            for _, contact in place_matches.iterrows():
                matched_contacts.append({
                    'contact': contact,
                    'match_type': 'Exact Google Place ID',
                    'match_score': 1.0,
                    'confidence': 'high',
                    'needs_verification': False
                })
        
        # Fuzzy name matching with improved logic
        for _, contact in contacts_df.iterrows():
            # Skip if already matched by Google Place ID
            if any(mc['contact']['Contact ID'] == contact['Contact ID'] and 
                   mc['match_type'] == 'Exact Google Place ID' for mc in matched_contacts):
                continue
            
            best_score = 0
            best_match_type = ''
            best_confidence = 'none'
            best_field = ''
            
            # Get contact city if available
            contact_city = contact.get('City', '') or contact.get('Macro Geo (NYC, SF, CHS, DC, LA, NASH, DEN)', '')
            
            # Check Deal Name
            if pd.notna(contact['Deal Name']):
                score, match_type, confidence = calculate_similarity_improved(
                    restaurant_name, 
                    contact['Deal Name'],
                    restaurant_city,
                    contact_city
                )
                if score > best_score:
                    best_score = score
                    best_match_type = match_type
                    best_confidence = confidence
                    best_field = 'Deal Name'
            
            # Check Company Name
            if pd.notna(contact['Company name']):
                score, match_type, confidence = calculate_similarity_improved(
                    restaurant_name,
                    contact['Company name'],
                    restaurant_city,
                    contact_city
                )
                if score > best_score:
                    best_score = score
                    best_match_type = match_type
                    best_confidence = confidence
                    best_field = 'Company Name'
            
            # Only add if score meets strict threshold
            if best_score >= strict_threshold:
                needs_verification = best_score < verify_threshold or best_confidence == 'low'
                
                matched_contacts.append({
                    'contact': contact,
                    'match_type': f'{best_match_type} ({best_field})',
                    'match_score': best_score,
                    'confidence': best_confidence,
                    'needs_verification': needs_verification
                })
                
                if needs_verification:
                    verification_needed.append({
                        'restaurant': restaurant_name,
                        'restaurant_city': restaurant_city,
                        'contact_name': contact.get('Deal Name') or contact.get('Company name'),
                        'score': best_score,
                        'confidence': best_confidence
                    })
        
        # Sort matches by score (highest first)
        matched_contacts.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Create output rows
        if not matched_contacts:
            results.append({
                'Restaurant': restaurant_name,
                'City': restaurant_city,
                'Google Places ID': google_place_id if pd.notna(google_place_id) else '',
                'TPV 90 Days': restaurant.get('TPV 90 Days', ''),
                'Red Flags': restaurant.get('Red Flags', ''),
                'Cohort': restaurant.get('Cohort', ''),
                'Match Status': 'No Matches Found',
                'Match Type': '',
                'Match Score': 0,
                'Match Confidence': '',
                'Needs Verification': '',
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
            for match in matched_contacts:
                contact = match['contact']
                results.append({
                    'Restaurant': restaurant_name,
                    'City': restaurant_city,
                    'Google Places ID': google_place_id if pd.notna(google_place_id) else '',
                    'TPV 90 Days': restaurant.get('TPV 90 Days', ''),
                    'Red Flags': restaurant.get('Red Flags', ''),
                    'Cohort': restaurant.get('Cohort', ''),
                    'Match Status': 'Matched',
                    'Match Type': match['match_type'],
                    'Match Score': f"{match['match_score']:.3f}",
                    'Match Confidence': match['confidence'],
                    'Needs Verification': 'Yes' if match['needs_verification'] else 'No',
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
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(restaurants_df)} restaurants...")
    
    return pd.DataFrame(results), verification_needed

def main():
    """
    Run the improved matching process.
    """
    print("=" * 60)
    print("IMPROVED Restaurant Contact Matcher")
    print("=" * 60)
    
    # File paths
    RESTAURANTS_FILE = "restaurantinfo.csv"
    CONTACTS_FILE = "contactinfo.csv"
    OUTPUT_FILE = "restaurant_contact_matches_improved.csv"
    
    try:
        # Load data
        print(f"\nLoading restaurant data from: {RESTAURANTS_FILE}")
        restaurants_df = pd.read_csv(RESTAURANTS_FILE)
        print(f"Loaded {len(restaurants_df)} restaurants")
        
        print(f"\nLoading contact data from: {CONTACTS_FILE}")
        contacts_df = pd.read_csv(CONTACTS_FILE)
        print(f"Loaded {len(contacts_df)} contacts")
        
        # Perform matching with better thresholds
        print("\n" + "=" * 60)
        print("Running Improved Matching Process")
        print("=" * 60)
        
        results_df, verification_needed = match_restaurants_with_validation(
            restaurants_df, 
            contacts_df,
            strict_threshold=0.65,  # Much higher than your 0.35
            verify_threshold=0.8    # High confidence matches don't need verification
        )
        
        # Save results
        results_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to: {OUTPUT_FILE}")
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("MATCHING SUMMARY")
        print("=" * 60)
        
        total_restaurants = len(restaurants_df)
        matched_restaurants = results_df[results_df['Match Status'] == 'Matched']['Restaurant'].nunique()
        no_match_restaurants = results_df[results_df['Match Status'] == 'No Matches Found']['Restaurant'].nunique()
        total_matches = len(results_df[results_df['Match Status'] == 'Matched'])
        
        print(f"Total restaurants: {total_restaurants}")
        print(f"Restaurants with matches: {matched_restaurants} ({matched_restaurants/total_restaurants*100:.1f}%)")
        print(f"Restaurants without matches: {no_match_restaurants}")
        print(f"Total matched contacts: {total_matches}")
        
        # Match type distribution
        print("\nMatch Type Distribution:")
        match_types = results_df[results_df['Match Status'] == 'Matched']['Match Type'].value_counts()
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count}")
        
        # Confidence distribution
        print("\nConfidence Distribution:")
        confidence_dist = results_df[results_df['Match Status'] == 'Matched']['Match Confidence'].value_counts()
        for conf, count in confidence_dist.items():
            print(f"  {conf}: {count}")
        
        # Verification needed
        needs_verify = results_df[
            (results_df['Match Status'] == 'Matched') & 
            (results_df['Needs Verification'] == 'Yes')
        ]
        print(f"\nMatches needing verification: {len(needs_verify)}")
        
        if len(verification_needed) > 0:
            print(f"\nTop 10 matches that might need Claude verification:")
            for i, item in enumerate(verification_needed[:10], 1):
                print(f"{i}. {item['restaurant']} <-> {item['contact_name']} (score: {item['score']:.3f}, confidence: {item['confidence']})")
        
        # Save verification candidates
        if verification_needed:
            verify_df = pd.DataFrame(verification_needed)
            verify_file = OUTPUT_FILE.replace('.csv', '_needs_verification.csv')
            verify_df.to_csv(verify_file, index=False)
            print(f"\nVerification candidates saved to: {verify_file}")
        
        print("\n✅ Matching complete!")
        print("\n💡 Next steps:")
        print("1. Review the high-confidence matches (confidence='high', score > 0.8)")
        print("2. Use Claude to verify only the low-confidence matches")
        print(f"3. This should reduce verification from 21,583 to ~{len(needs_verify)} matches")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()