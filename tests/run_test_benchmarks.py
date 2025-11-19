#!/usr/bin/env python3
"""
Run Test Benchmarks - SMALL VALIDATION TEST

Quick validation test with just 10 samples per benchmark to verify fixes work.

Cost: ~$5 | Time: ~10-15 minutes

Usage:
  export LLM_API_KEY='your-key-here'
  python3 run_test_benchmarks.py
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Check API key
api_key = os.getenv('LLM_API_KEY')
if not api_key:
    print("Error: LLM_API_KEY not set!")
    print("\nSet your API key:")
    print("  set LLM_API_KEY=your-key-here  (Windows)")
    print("  export LLM_API_KEY='your-key-here'  (Linux/Mac)")
    sys.exit(1)

print("=" * 80)
print("QUICK VALIDATION TEST - 10 SAMPLES PER BENCHMARK")
print("=" * 80)
print(f"\n✓ API Key: {api_key[:20]}...")
print(f"✓ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Benchmark configuration - SMALL TESTS
benchmarks = [
    {
        "name": "Text Masking (ai4privacy)",
        "script": "benchmarks/public_datasets_simple.py",
        "args": ["--num-samples", "10"],
        "total_samples": 10,
    },
    {
        "name": "PUPA (Question Answering)",
        "script": "benchmarks/pupa_benchmark.py",
        "args": ["--simulate", "--num-samples", "10"],
        "total_samples": 10,
    },
    {
        "name": "TAB (Document Sanitization)",
        "script": "benchmarks/text_sanitization_benchmark.py",
        "args": ["--simulate", "--num-samples", "10"],
        "total_samples": 10,
    }
]

total_samples = sum(b['total_samples'] for b in benchmarks)

print("\n" + "=" * 80)
print("BENCHMARK PLAN")
print("=" * 80)

for i, bench in enumerate(benchmarks, 1):
    print(f"\n{i}. {bench['name']}")
    print(f"   Samples: {bench['total_samples']}")

print("\n" + "=" * 80)
print(f"TOTAL SAMPLES: {total_samples}")
print(f"ESTIMATED COST: ~$5")
print(f"ESTIMATED TIME: ~10-15 minutes")
print("=" * 80)

print("\n📊 This is a QUICK VALIDATION TEST to verify ensemble fixes work.")
print("   Look for: '📊 Ensemble (4 models): PII leaked = [...]' in output")

response = input("\nProceed with validation test? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("\n❌ Test cancelled.")
    sys.exit(0)

# Run benchmarks
results = []

print("\n" + "=" * 80)
print("RUNNING VALIDATION TESTS")
print("=" * 80)

for i, bench in enumerate(benchmarks, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {i}/{len(benchmarks)}: {bench['name']}")
    print(f"{'=' * 80}")

    start_time = time.time()

    cmd = ["python3", bench['script']] + bench['args']
    print(f"\n▶ Running: {' '.join(cmd)}")

    try:
        env = os.environ.copy()
        env['LLM_API_KEY'] = api_key
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )

        ensemble_found = False
        for line in process.stdout:
            print(line, end='')
            if '📊 Ensemble' in line:
                ensemble_found = True

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            status = "✅ SUCCESS"
            if ensemble_found:
                status += " (Ensemble working!)"
        else:
            status = "❌ FAILED"

        print(f"\n{status} ({elapsed:.1f}s)")

        results.append({
            'name': bench['name'],
            'status': 'success' if process.returncode == 0 else 'failed',
            'ensemble_found': ensemble_found,
            'elapsed': elapsed
        })

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        results.append({
            'name': bench['name'],
            'status': 'error',
            'error': str(e)
        })

# Summary
print("\n" + "=" * 80)
print("VALIDATION TEST COMPLETE")
print("=" * 80)

successes = sum(1 for r in results if r['status'] == 'success')
ensemble_working = sum(1 for r in results if r.get('ensemble_found', False))

print(f"\n✓ Successful: {successes}/{len(benchmarks)}")
print(f"✓ Ensemble detected: {ensemble_working}/{len(benchmarks)}")

if ensemble_working == len(benchmarks):
    print("\n🎉 SUCCESS! Ensemble consensus is working!")
    print("\nNext steps:")
    print("  1. Run medium-scale test (100 samples): modify run_all_benchmarks.py")
    print("  2. Or edit and run full benchmarks if confident")
elif ensemble_working > 0:
    print("\n⚠️  PARTIAL SUCCESS - Ensemble working in some benchmarks")
    print("   Check which ones failed above")
else:
    print("\n❌ ENSEMBLE NOT DETECTED!")
    print("   The fixes may not be working correctly")
    print("   Check the output above for '📊 Ensemble' messages")

print("\n" + "=" * 80)
