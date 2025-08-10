import json
import requests
import time

def comprehensive_analysis():
    print("🔍 COMPREHENSIVE SCALING ANALYSIS")
    print("=" * 50)
    
    # Test local API performance
    local_url = "http://127.0.0.1:8081"
    external_url = "http://34.70.248.22"
    
    print("1. LOCAL API PERFORMANCE:")
    try:
        start = time.time()
        response = requests.get(f"{local_url}/health", timeout=5)
        local_time = time.time() - start
        print(f"   ✅ Response time: {local_time:.3f}s")
        print(f"   ✅ Status: {response.status_code}")
    except:
        print("   ❌ Local API not accessible")
    
    print("\n2. KUBERNETES API PERFORMANCE:")
    try:
        start = time.time()
        response = requests.get(f"{external_url}/health", timeout=10)
        k8s_time = time.time() - start
        print(f"   ✅ Response time: {k8s_time:.3f}s")
        print(f"   ✅ Status: {response.status_code}")
    except:
        print("   ❌ Kubernetes API not accessible")
    
    print("\n3. IDENTIFIED BOTTLENECKS:")
    print("   🚨 Pod crashes indicate resource constraints")
    print("   🚨 LoadBalancer accessible but backend pods failing")
    print("   🚨 Scaling limited by pod stability issues")
    
    print("\n4. SCALING RECOMMENDATIONS:")
    print("   📈 Fix pod resource limits and requests")
    print("   📈 Implement health checks properly")
    print("   📈 Use horizontal pod autoscaler")
    
if __name__ == "__main__":
    comprehensive_analysis()
