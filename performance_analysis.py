import json
import matplotlib.pyplot as plt
import numpy as np

# Analyze results if load_test_results.json exists
try:
    with open('load_test_results.json', 'r') as f:
        results = json.load(f)
    
    print("📊 BOTTLENECK ANALYSIS")
    print("=" * 40)
    
    successful = [r for r in results if r['success']]
    response_times = [r['response_time'] for r in successful]
    
    if response_times:
        print(f"Requests processed: {len(results)}")
        print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        print(f"Average response time: {np.mean(response_times):.3f}s")
        print(f"95th percentile: {np.percentile(response_times, 95):.3f}s")
        
        # Identify bottlenecks
        if np.mean(response_times) > 0.5:
            print("🚨 BOTTLENECK: High latency detected")
        if np.std(response_times) > 0.2:
            print("🚨 BOTTLENECK: High variance in response times")
        if len(successful) < len(results):
            print(f"🚨 BOTTLENECK: {len(results) - len(successful)} failed requests")
            
except FileNotFoundError:
    print("No load test results found. Run load_test.py first.")
