#!/usr/bin/env python3
"""
Quick API Connection Test

Tests if LLM API is accessible and working.
"""

import os
import sys

# Check API key
api_key = os.getenv('LLM_API_KEY')
if not api_key:
    print("❌ LLM_API_KEY not set!")
    print("\nRun: export LLM_API_KEY='your-key-here'")
    sys.exit(1)

print(f"✓ API Key found: {api_key[:20]}...")

# Test LLM SDK import
print("\n1. Testing LLM SDK import...")
try:
    from sambanova import SambaNova
    print("   ✓ LLM SDK imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import LLM SDK: {e}")
    print("   Run: pip install sambanova")
    sys.exit(1)

# Test client initialization
print("\n2. Initializing API client...")
try:
    client = SambaNova(
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1"
    )
    print("   ✓ Client initialized successfully")
except Exception as e:
    print(f"   ❌ Failed to initialize client: {e}")
    sys.exit(1)

# Test simple API call
print("\n3. Testing API call (simple completion)...")
try:
    response = client.chat.completions.create(
        model="Meta-Llama-3.1-8B-Instruct",  # Fast, small model for testing
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API connection successful' if you can read this."}
        ],
        max_tokens=50,
        temperature=0.1
    )

    result = response.choices[0].message.content.strip()
    print(f"   ✓ API call successful!")
    print(f"   Response: {result}")

except Exception as e:
    print(f"   ❌ API call failed: {e}")
    print(f"\n   Error details: {type(e).__name__}: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - API is working correctly!")
print("="*80)
print("\nYou can now run the benchmarks:")
print("  python3 run_all_benchmarks.py")
