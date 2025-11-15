#!/usr/bin/env python3
"""
Run All Benchmarks - Comprehensive Evaluation Suite (FULL DATASETS)

This script runs ALL available benchmarks with MAXIMUM samples from each dataset
to provide a complete, thorough evaluation of the ensemble-redaction privacy pipeline.

⚠️  WARNING: This is a COMPREHENSIVE evaluation using ALL available data samples.
    Estimated cost: $300-380 | Estimated time: 7-9 hours

Usage:
  export LLM_API_KEY='your-key-here'
  python3 run_all_benchmarks.py

Benchmarks included (using public datasets and vendor-neutral formats):
1. Vendor-Neutral Synthetic - 300 samples (100 per domain: medical, financial, education)
2. ai4privacy/pii-masking-200k - 1,000 samples (from 200K+ available, 54 PII types)
3. PUPA (NAACL 2025) - ALL 901 real user-agent interactions from WildChat
4. TAB - ALL 1,268 ECHR court cases with manual PII annotations
5. Differential Privacy - 100 samples (Canary exposure, MIA, DP comparison)

Total: 3,569 samples across 5 benchmarks

Note: run_benchmarks.py and run_demo_pipeline.py use legacy Microsoft-specific
      field names (MSNClicks, BingSearch) for demo purposes only and are NOT
      included in this comprehensive evaluation suite.

Individual benchmark estimates:
- Vendor-Neutral: 300 samples, ~60-75 min, ~$40-50
- ai4privacy: 1,000 samples, ~120-150 min, ~$80-100
- PUPA: 901 samples, ~90-120 min, ~$60-80
- TAB: 1,268 samples, ~120-150 min, ~$80-100
- DP Comparison: 100 samples, ~60-75 min, ~$40-50
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# Check API key
api_key = os.getenv('LLM_API_KEY')
if not api_key:
    print("❌ Error: LLM_API_KEY not set!")
    print("\nSet your API key:")
    print("  export LLM_API_KEY='your-key-here'")
    sys.exit(1)

print("=" * 80)
print("COMPREHENSIVE BENCHMARK SUITE - PUBLIC DATASETS & VENDOR-NEUTRAL FORMATS")
print("=" * 80)
print(f"\n✓ API Key: {api_key[:20]}...")
print(f"✓ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Benchmark configuration
# Note: run_benchmarks.py and run_demo_pipeline.py use legacy Microsoft-specific
# field names (MSNClicks, BingSearch) for demo purposes only. Real benchmarks
# use standard formats from public datasets.

benchmarks = [
    {
        "name": "Vendor-Neutral Synthetic Benchmark",
        "script": "benchmarks/neutral_benchmark.py",
        "args": ["--benchmark", "all", "--domains", "all", "--num-samples", "100"],
        "description": "Synthetic data with vendor-neutral field names - 100 samples per domain (300 total)",
        "dataset_type": "Synthetic",
        "total_samples": 300,
        "estimated_time": "60-75 min",
        "estimated_cost": "$40-50"
    },
    {
        "name": "ai4privacy/pii-masking-200k",
        "script": "benchmarks/public_datasets_simple.py",
        "args": ["--num-samples", "1000"],
        "description": "Real PII dataset from Hugging Face - 1,000 samples (from 200K+ available)",
        "dataset_type": "Public Dataset",
        "total_samples": 1000,
        "estimated_time": "120-150 min",
        "estimated_cost": "$80-100"
    },
    {
        "name": "PUPA (NAACL 2025)",
        "script": "benchmarks/pupa_benchmark.py",
        "args": ["--simulate", "--num-samples", "901"],
        "description": "Private User Prompt Annotations - ALL 901 real user-agent interactions",
        "dataset_type": "Public Dataset",
        "total_samples": 901,
        "estimated_time": "90-120 min",
        "estimated_cost": "$60-80"
    },
    {
        "name": "TAB - Text Anonymization Benchmark",
        "script": "benchmarks/text_sanitization_benchmark.py",
        "args": ["--simulate", "--num-samples", "1268"],
        "description": "ECHR court cases - ALL 1,268 cases with manual PII annotations",
        "dataset_type": "Public Dataset",
        "total_samples": 1268,
        "estimated_time": "120-150 min",
        "estimated_cost": "$80-100"
    },
    {
        "name": "Differential Privacy Comparison",
        "script": "benchmarks/dp_benchmark.py",
        "args": ["--num-samples", "100"],
        "description": "Canary exposure, MIA, and DP (ε=1.0, ε=5.0) comparison - 100 samples",
        "dataset_type": "Synthetic",
        "total_samples": 100,
        "estimated_time": "60-75 min",
        "estimated_cost": "$40-50"
    }
]

# Summary
print("\n" + "=" * 80)
print("BENCHMARK PLAN")
print("=" * 80)

total_estimated_cost_min = 0
total_estimated_cost_max = 0
total_estimated_time_min = 0
total_estimated_time_max = 0

total_samples = sum(bench['total_samples'] for bench in benchmarks)

for i, bench in enumerate(benchmarks, 1):
    print(f"\n{i}. {bench['name']}")
    print(f"   Script: {bench['script']}")
    print(f"   Samples: {bench['total_samples']}")
    print(f"   Description: {bench['description']}")
    print(f"   Estimated time: {bench['estimated_time']}")
    print(f"   Estimated cost: {bench['estimated_cost']}")

    # Parse cost range
    cost_parts = bench['estimated_cost'].replace('$', '').split('-')
    total_estimated_cost_min += int(cost_parts[0])
    total_estimated_cost_max += int(cost_parts[1])

    # Parse time range (convert to minutes)
    time_parts = bench['estimated_time'].split('-')
    total_estimated_time_min += int(time_parts[0].split()[0])
    total_estimated_time_max += int(time_parts[1].split()[0])

print("\n" + "=" * 80)
print(f"TOTAL SAMPLES: {total_samples:,}")
print(f"TOTAL ESTIMATED COST: ${total_estimated_cost_min}-${total_estimated_cost_max}")
print(f"TOTAL ESTIMATED TIME: {total_estimated_time_min//60}h {total_estimated_time_min%60}m - {total_estimated_time_max//60}h {total_estimated_time_max%60}m")
print("=" * 80)

# Confirm
print("\n⚠️  WARNING: FULL DATASET EVALUATION")
print(f"   This will process {total_samples:,} samples across ALL benchmarks.")
print("   This is a COMPREHENSIVE evaluation using ALL available data.")
print(f"   Estimated cost: ${total_estimated_cost_min}-${total_estimated_cost_max}")
print(f"   Estimated time: {total_estimated_time_min//60}h {total_estimated_time_min%60}m - {total_estimated_time_max//60}h {total_estimated_time_max%60}m")
print("   Make sure you have:")
print("   - Stable internet connection")
print("   - Sufficient API credits")
print("   - Time to complete (7-9 hours)")

response = input("\n📊 Proceed with full benchmark suite? (yes/no): ")
if response.lower() not in ['yes', 'y']:
    print("\n❌ Benchmark suite cancelled.")
    sys.exit(0)

# Run benchmarks
results_summary = {
    "start_time": datetime.now().isoformat(),
    "benchmarks": []
}

print("\n" + "=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)

for i, bench in enumerate(benchmarks, 1):
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK {i}/{len(benchmarks)}: {bench['name']}")
    print(f"{'=' * 80}")

    start_time = time.time()

    cmd = ["python3", bench['script']] + bench['args']
    print(f"\n▶ Running: {' '.join(cmd)}")
    print(f"⏱  Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🔑 API Key available: {api_key[:20]}... (length: {len(api_key)})")

    try:
        # Stream output in real-time while also saving to log
        log_file = f"{bench['script'].replace('/', '_').replace('.py', '')}_log.txt"

        import subprocess as sp
        import select

        # Create environment with API key explicitly set
        env = os.environ.copy()
        env['LLM_API_KEY'] = api_key  # Ensure API key is passed
        env['PYTHONUNBUFFERED'] = '1'  # Disable Python output buffering

        print(f"[DEBUG] Starting subprocess with unbuffered output...", flush=True)
        process = sp.Popen(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            bufsize=1,
            env=env  # Pass environment variables to subprocess
        )
        print(f"[DEBUG] Subprocess started with PID: {process.pid}", flush=True)

        output_lines = []
        last_output_time = time.time()
        with open(log_file, 'w') as log:
            try:
                print(f"[DEBUG] Entering output reading loop...", flush=True)
                line_count = 0
                for line in process.stdout:
                    line_count += 1
                    print(line, end='')  # Real-time output to console
                    log.write(line)      # Save to log file
                    output_lines.append(line)
                    log.flush()
                    last_output_time = time.time()

                    # Debug heartbeat every 10 lines
                    if line_count % 10 == 0:
                        print(f"[DEBUG] Read {line_count} lines so far...", flush=True)

                print(f"[DEBUG] Finished reading output ({line_count} total lines)", flush=True)
                print(f"[DEBUG] Waiting for process to complete...", flush=True)
                process.wait(timeout=10800)  # 3 hour timeout
                print(f"[DEBUG] Process completed with return code: {process.returncode}", flush=True)

            except sp.TimeoutExpired:
                print(f"[DEBUG] Process timeout after 3 hours", flush=True)
                process.kill()
                raise

        elapsed_time = time.time() - start_time
        result_stdout = ''.join(output_lines)

        if process.returncode == 0:
            print(f"\n✅ SUCCESS ({elapsed_time:.1f}s)")
            print(f"   Log saved to: {log_file}")

            # Try to extract results file
            results_file = None
            for line in output_lines:
                if 'saved to:' in line.lower() or 'results:' in line.lower():
                    parts = line.split()
                    if any(p.endswith('.json') for p in parts):
                        results_file = next(p for p in parts if p.endswith('.json'))
                        break

            benchmark_result = {
                "name": bench['name'],
                "script": bench['script'],
                "status": "success",
                "elapsed_time": elapsed_time,
                "results_file": results_file,
                "log_file": log_file
            }

        else:
            print(f"\n❌ FAILED ({elapsed_time:.1f}s)")
            print(f"   Check log: {log_file}")

            benchmark_result = {
                "name": bench['name'],
                "script": bench['script'],
                "status": "failed",
                "elapsed_time": elapsed_time,
                "log_file": log_file
            }

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  TIMEOUT ({elapsed_time:.1f}s)")

        benchmark_result = {
            "name": bench['name'],
            "script": bench['script'],
            "status": "timeout",
            "elapsed_time": elapsed_time
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ ERROR ({elapsed_time:.1f}s): {e}")

        benchmark_result = {
            "name": bench['name'],
            "script": bench['script'],
            "status": "error",
            "elapsed_time": elapsed_time,
            "error": str(e)
        }

    results_summary["benchmarks"].append(benchmark_result)

    print(f"\n⏱  Completed at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⏱  Time taken: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")

# Final summary
results_summary["end_time"] = datetime.now().isoformat()
total_elapsed = sum(b["elapsed_time"] for b in results_summary["benchmarks"])
results_summary["total_elapsed_time"] = total_elapsed

# Save summary
summary_file = "benchmark_suite_summary.json"
with open(summary_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 80)
print("BENCHMARK SUITE COMPLETE")
print("=" * 80)

print(f"\n✓ Total time: {total_elapsed//3600:.0f}h {(total_elapsed%3600)//60:.0f}m {total_elapsed%60:.0f}s")
print(f"✓ Benchmarks run: {len(benchmarks)}")

# Count successes/failures
successes = sum(1 for b in results_summary["benchmarks"] if b["status"] == "success")
failures = sum(1 for b in results_summary["benchmarks"] if b["status"] != "success")

print(f"✓ Successful: {successes}/{len(benchmarks)}")
if failures > 0:
    print(f"⚠️  Failed: {failures}/{len(benchmarks)}")

print(f"\n✓ Summary saved to: {summary_file}")

# Show results
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

for i, bench in enumerate(results_summary["benchmarks"], 1):
    status_emoji = "✅" if bench["status"] == "success" else "❌"
    elapsed = bench["elapsed_time"]
    print(f"\n{i}. {status_emoji} {bench['name']}")
    print(f"   Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"   Status: {bench['status']}")
    if bench.get("results_file"):
        print(f"   Results: {bench['results_file']}")

print("\n" + "=" * 80)
print("\n🎉 All benchmarks completed!")
print(f"📊 Check {summary_file} for detailed results")
print("\n" + "=" * 80)
