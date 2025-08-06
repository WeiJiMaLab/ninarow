#!/usr/bin/env python3
"""
Performance test script to compare optimized vs original tree search performance.
"""
import subprocess
import sys
import time

def run_timing_test():
    """Run the optimized timing test."""
    print("=" * 60)
    print("RUNNING OPTIMIZED TREE SEARCH PERFORMANCE TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, "model_fitting/tree_search_bads_timing.py"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✅ OPTIMIZED VERSION COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total execution time: {elapsed_time:.2f}s")
            print("\n" + "="*40 + " OUTPUT " + "="*40)
            print(result.stdout)
            if result.stderr:
                print("\n" + "="*40 + " STDERR " + "="*40)
                print(result.stderr)
        else:
            print("❌ OPTIMIZED VERSION FAILED!")
            print(f"Exit code: {result.returncode}")
            print(f"Execution time before failure: {elapsed_time:.2f}s")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏰ OPTIMIZED VERSION TIMED OUT after {elapsed_time:.2f}s")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 OPTIMIZED VERSION ERROR after {elapsed_time:.2f}s: {e}")

    print("\n" + "=" * 60)
    print("PERFORMANCE TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_timing_test()