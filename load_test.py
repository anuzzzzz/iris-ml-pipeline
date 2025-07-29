import requests
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np

class LoadTester:
    def __init__(self, base_url="http://127.0.0.1:8081"):
        self.base_url = base_url
        self.results = []
        
    def single_prediction(self, request_id):
        """Single prediction request"""
        start_time = time.time()
        try:
            data = {
                "features": [5.1 + np.random.normal(0, 0.1), 
                           3.5 + np.random.normal(0, 0.1),
                           1.4 + np.random.normal(0, 0.1), 
                           0.2 + np.random.normal(0, 0.1)]
            }
            response = requests.post(f"{self.base_url}/predict", 
                                   json=data, timeout=30)
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'success': response.status_code == 200,
                'timestamp': start_time
            }
        except Exception as e:
            end_time = time.time()
            return {
                'request_id': request_id,
                'status_code': 0,
                'response_time': end_time - start_time,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            }
    
    def run_load_test(self, num_requests=50, max_workers=10):
        """Run concurrent load test"""
        print(f"🚀 Starting load test: {num_requests} requests with {max_workers} workers")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.single_prediction, i) 
                      for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                
                if len(self.results) % 10 == 0:
                    print(f"Completed {len(self.results)}/{num_requests} requests")
        
        total_time = time.time() - start_time
        self.analyze_results(total_time)
        
    def analyze_results(self, total_time):
        """Analyze load test results"""
        successful_requests = [r for r in self.results if r['success']]
        failed_requests = [r for r in self.results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            print("\n📊 LOAD TEST RESULTS")
            print("=" * 50)
            print(f"Total Requests: {len(self.results)}")
            print(f"Successful: {len(successful_requests)}")
            print(f"Failed: {len(failed_requests)}")
            print(f"Success Rate: {len(successful_requests)/len(self.results)*100:.1f}%")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"Requests/Second: {len(self.results)/total_time:.2f}")
            print(f"Avg Response Time: {np.mean(response_times):.3f}s")
            print(f"Min Response Time: {np.min(response_times):.3f}s")
            print(f"Max Response Time: {np.max(response_times):.3f}s")
            print(f"95th Percentile: {np.percentile(response_times, 95):.3f}s")
            
            # Identify bottlenecks
            print("\n🔍 BOTTLENECK ANALYSIS")
            print("=" * 50)
            if np.mean(response_times) > 1.0:
                print("⚠️ HIGH LATENCY: Average response time > 1 second")
            if len(failed_requests) > 0:
                print(f"⚠️ FAILED REQUESTS: {len(failed_requests)} requests failed")
            if np.max(response_times) > 5.0:
                print("⚠️ TIMEOUT ISSUES: Some requests took > 5 seconds")
                
        # Save results
        with open('load_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📄 Results saved to load_test_results.json")

if __name__ == "__main__":
    tester = LoadTester()
    
    # Test with increasing load
    print("Phase 1: Baseline test (10 requests, 2 workers)")
    tester.run_load_test(num_requests=10, max_workers=2)
    
    time.sleep(2)
    tester.results = []
    
    print("\nPhase 2: Medium load (50 requests, 10 workers)")
    tester.run_load_test(num_requests=50, max_workers=10)
    
    time.sleep(2)
    tester.results = []
    
    print("\nPhase 3: High load (100 requests, 20 workers)")
    tester.run_load_test(num_requests=100, max_workers=20)
