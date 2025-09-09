#!/usr/bin/env python3
"""
Test Script for Claude Verification - Tests first 10 unique pairings only
This script tests the Claude verification functionality with a small subset of data
to debug issues before running the full verification.
"""

import pandas as pd
import numpy as np
import json
from anthropic import Anthropic
import time
import os
import sys

def test_claude_verification(restaurants_file="restaurantinfo.csv", 
                            contacts_file="contactinfo.csv",
                            max_pairings=10):
    """
    Test Claude verification with a limited number of pairings.
    """
    print("=" * 60)
    print("CLAUDE VERIFICATION TEST SCRIPT")
    print(f"Testing with first {max_pairings} unique pairings")
    print("=" * 60)
    
    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not found in environment variables.")
        api_key = input("Please enter your Anthropic API key: ").strip()
        if not api_key:
            print("❌ API key is required. Exiting.")
            return
    
    # Initialize Anthropic client
    print("\nInitializing Anthropic client...")
    client = Anthropic(api_key=api_key)
    
    # Load the pre-verification results if they exist
    pre_verify_file = "restaurant_contact_matches_pre_verification.csv"
    
    if os.path.exists(pre_verify_file):
        print(f"\nLoading existing pre-verification results from: {pre_verify_file}")
        results_df = pd.read_csv(pre_verify_file)
    else:
        print("\n❌ Pre-verification file not found. Please run the main script first to generate matches.")
        return
    
    # Get fuzzy matches only
    fuzzy_matches = results_df[
        (results_df['Match Status'] == 'Matched') & 
        (results_df['Match Type'] != 'Exact Google Place ID')
    ].copy()
    
    print(f"\nFound {len(fuzzy_matches)} total fuzzy matches")
    
    # Get unique pairings
    unique_pairings = fuzzy_matches[['Restaurant', 'City', 'Contact Deal Name', 'Contact Company']].drop_duplicates()
    unique_pairings = unique_pairings[unique_pairings['Contact Deal Name'].notna() & (unique_pairings['Contact Deal Name'] != '')]
    
    print(f"Found {len(unique_pairings)} unique restaurant-contact pairings")
    
    # Limit to first N pairings for testing
    test_pairings = unique_pairings.head(max_pairings)
    print(f"\n🧪 Testing with {len(test_pairings)} pairings:")
    print("-" * 60)
    
    for i, (_, row) in enumerate(test_pairings.iterrows(), 1):
        print(f"{i}. Restaurant: {row['Restaurant']}")
        print(f"   Deal Name: {row['Contact Deal Name']}")
        print()
    
    # Prepare batch for Claude
    batch_items = []
    for _, row in test_pairings.iterrows():
        item = {
            "restaurant_name": row['Restaurant'],
            "restaurant_city": row['City'] if pd.notna(row['City']) else "Unknown",
            "contact_deal_name": row['Contact Deal Name'],
            "contact_company": row['Contact Company'] if pd.notna(row['Contact Company']) else ""
        }
        batch_items.append(item)
    
    print("=" * 60)
    print("TESTING CLAUDE API CALL")
    print("=" * 60)
    
    # Create the prompts
    system_prompt = """You are a restaurant verification expert. Your task is to determine if restaurant names and contact deal names actually refer to the SAME physical establishment or business.

For each pairing, consider:
1. Are these clearly the same business (even if names differ slightly)?
2. Could one be a parent company/franchise and the other a specific location?
3. Are there any indicators these might be DIFFERENT businesses?

Respond with a JSON array where each item has:
- "index": the index number (0-based)
- "same_place": true/false (true if they refer to the same establishment)
- "confidence": "high", "medium", or "low"
- "reason": brief explanation (max 50 words)

Return ONLY valid JSON, no other text or markdown."""

    user_prompt = f"""Please verify if these restaurant names and contact deal names refer to the same establishments:

{json.dumps(batch_items, indent=2)}"""

    print("\n📤 Sending request to Claude API...")
    print(f"   Model: claude-3-5-sonnet-20241022")
    print(f"   Number of pairings: {len(batch_items)}")
    
    try:
        # Call Claude API without web search tool for simpler testing
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        print("\n✅ API call successful!")
        
        # Get the response text
        response_text = response.content[0].text.strip()
        
        print("\n📥 Raw response from Claude:")
        print("-" * 60)
        print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        print("-" * 60)
        
        # Try to parse JSON
        print("\n🔄 Attempting to parse JSON response...")
        
        # Clean up response if needed
        cleaned_response = response_text
        
        # Remove markdown code blocks if present
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[1].split("```")[0]
        elif "```" in cleaned_response:
            parts = cleaned_response.split("```")
            if len(parts) >= 2:
                cleaned_response = parts[1]
        
        # Parse JSON
        verification_results = json.loads(cleaned_response)
        
        print(f"✅ Successfully parsed {len(verification_results)} results")
        
        # Display results
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        for j, result in enumerate(verification_results):
            if j < len(test_pairings):
                row = test_pairings.iloc[j]
                status = "✅ SAME" if result['same_place'] else "❌ DIFFERENT"
                print(f"\n{j+1}. {status} (Confidence: {result['confidence']})")
                print(f"   Restaurant: {row['Restaurant']}")
                print(f"   Deal Name: {row['Contact Deal Name']}")
                print(f"   Reason: {result['reason']}")
        
        # Statistics
        same_count = sum(1 for r in verification_results if r['same_place'])
        different_count = len(verification_results) - same_count
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total verified: {len(verification_results)}")
        print(f"Same place: {same_count} ({same_count/len(verification_results)*100:.1f}%)")
        print(f"Different: {different_count} ({different_count/len(verification_results)*100:.1f}%)")
        
        # Confidence distribution
        confidence_counts = {}
        for r in verification_results:
            conf = r.get('confidence', 'unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        print("\nConfidence distribution:")
        for conf, count in confidence_counts.items():
            print(f"  {conf}: {count}")
        
        print("\n✅ Test completed successfully!")
        print("\n💡 Next steps:")
        print("1. Review the results above to ensure they make sense")
        print("2. If results look good, you can run the full verification")
        print("3. Consider adding web search for better accuracy (adds complexity)")
        
        # Save test results
        test_output_file = "test_verification_results.json"
        with open(test_output_file, 'w') as f:
            json.dump({
                "test_pairings": batch_items,
                "verification_results": verification_results,
                "summary": {
                    "total": len(verification_results),
                    "same_place": same_count,
                    "different": different_count,
                    "confidence_distribution": confidence_counts
                }
            }, f, indent=2)
        
        print(f"\n💾 Test results saved to: {test_output_file}")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON response: {e}")
        print("\nResponse text that failed to parse:")
        print("-" * 60)
        print(response_text if 'response_text' in locals() else "No response text available")
        print("-" * 60)
        print("\n💡 The model may not be returning valid JSON. Check the response format above.")
        
    except Exception as e:
        print(f"\n❌ Error during API call: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # More detailed error information
        if hasattr(e, 'response'):
            print(f"   Response status: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"   Response text: {getattr(e.response, 'text', 'N/A')}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

def main():
    """
    Main entry point for the test script.
    """
    # Check command line arguments
    max_pairings = 10
    
    if len(sys.argv) > 1:
        try:
            max_pairings = int(sys.argv[1])
            print(f"Using {max_pairings} pairings from command line argument")
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, using default of 10")
    
    test_claude_verification(max_pairings=max_pairings)

if __name__ == "__main__":
    main()