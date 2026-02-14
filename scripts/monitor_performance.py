#!/usr/bin/env python3
"""
Performance monitoring script for CPU-optimized Contact Center AI
Monitors system resources and provides optimization recommendations
"""

import psutil
import time
import requests
import json
from datetime import datetime
import argparse

class PerformanceMonitor:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.start_time = time.time()
        
    def get_system_info(self):
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def test_api_performance(self):
        """Test API endpoint performance."""
        endpoints = [
            '/api/v1/health',
            '/api/v1/conversations/analyze',
            '/api/v1/rag/query'
        ]
        
        results = {}
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                
                if endpoint == '/api/v1/health':
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                elif endpoint == '/api/v1/conversations/analyze':
                    payload = {
                        "customer_message": "I need help with my order",
                        "agent_response": "I'll help you with that right away"
                    }
                    response = requests.post(f"{self.api_url}{endpoint}", 
                                           json=payload, timeout=10)
                elif endpoint == '/api/v1/rag/query':
                    payload = {"question": "What is your refund policy?"}
                    response = requests.post(f"{self.api_url}{endpoint}", 
                                           json=payload, timeout=10)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                results[endpoint] = {
                    'response_time_ms': round(response_time, 2),
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                }
                
            except Exception as e:
                results[endpoint] = {
                    'response_time_ms': None,
                    'status_code': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_recommendations(self, system_info, api_results):
        """Generate performance recommendations."""
        recommendations = []
        
        # CPU recommendations
        if system_info['cpu_percent'] > 80:
            recommendations.append({
                'type': 'warning',
                'message': f"High CPU usage ({system_info['cpu_percent']:.1f}%). Consider reducing concurrent requests or using OpenAI API for heavy tasks."
            })
        elif system_info['cpu_percent'] > 60:
            recommendations.append({
                'type': 'info',
                'message': f"Moderate CPU usage ({system_info['cpu_percent']:.1f}%). System is performing well."
            })
        
        # Memory recommendations
        if system_info['memory_percent'] > 85:
            recommendations.append({
                'type': 'warning',
                'message': f"High memory usage ({system_info['memory_percent']:.1f}%). Consider reducing cache size or restarting the application."
            })
        elif system_info['memory_available_gb'] < 2:
            recommendations.append({
                'type': 'warning',
                'message': f"Low available memory ({system_info['memory_available_gb']:.1f}GB). Close other applications or increase swap space."
            })
        
        # API performance recommendations
        for endpoint, result in api_results.items():
            if result['success'] and result['response_time_ms']:
                if result['response_time_ms'] > 2000:
                    recommendations.append({
                        'type': 'warning',
                        'message': f"Slow response from {endpoint} ({result['response_time_ms']:.0f}ms). Consider optimizing or using external APIs."
                    })
                elif result['response_time_ms'] > 1000:
                    recommendations.append({
                        'type': 'info',
                        'message': f"Moderate response time from {endpoint} ({result['response_time_ms']:.0f}ms)."
                    })
            elif not result['success']:
                recommendations.append({
                    'type': 'error',
                    'message': f"Failed to connect to {endpoint}. Check if the API server is running."
                })
        
        # General recommendations for low-spec systems
        if system_info['memory_used_gb'] > 8:
            recommendations.append({
                'type': 'tip',
                'message': "For better performance on low-spec systems: Enable caching, use smaller batch sizes, consider OpenAI API for complex tasks."
            })
        
        return recommendations
    
    def run_monitoring(self, duration_minutes=5, interval_seconds=30):
        """Run continuous monitoring for specified duration."""
        print(f"🔍 Starting performance monitoring for {duration_minutes} minutes...")
        print(f"📊 Collecting data every {interval_seconds} seconds")
        print("-" * 60)
        
        end_time = time.time() + (duration_minutes * 60)
        data_points = []
        
        while time.time() < end_time:
            # Collect system info
            system_info = self.get_system_info()
            
            # Test API performance
            api_results = self.test_api_performance()
            
            # Combine data
            data_point = {
                'system': system_info,
                'api': api_results
            }
            data_points.append(data_point)
            
            # Print current status
            print(f"⏰ {system_info['timestamp']}")
            print(f"💻 CPU: {system_info['cpu_percent']:.1f}% | "
                  f"RAM: {system_info['memory_percent']:.1f}% "
                  f"({system_info['memory_used_gb']:.1f}GB used)")
            
            # Print API response times
            for endpoint, result in api_results.items():
                if result['success']:
                    print(f"🌐 {endpoint}: {result['response_time_ms']:.0f}ms")
                else:
                    print(f"❌ {endpoint}: Failed")
            
            print("-" * 60)
            
            time.sleep(interval_seconds)
        
        # Generate final report
        self.generate_report(data_points)
    
    def generate_report(self, data_points):
        """Generate performance report."""
        if not data_points:
            print("❌ No data collected")
            return
        
        print("\n📋 PERFORMANCE REPORT")
        print("=" * 60)
        
        # Calculate averages
        avg_cpu = sum(dp['system']['cpu_percent'] for dp in data_points) / len(data_points)
        avg_memory = sum(dp['system']['memory_percent'] for dp in data_points) / len(data_points)
        
        print(f"📊 Average CPU Usage: {avg_cpu:.1f}%")
        print(f"📊 Average Memory Usage: {avg_memory:.1f}%")
        
        # API performance summary
        print("\n🌐 API Performance Summary:")
        endpoint_stats = {}
        
        for data_point in data_points:
            for endpoint, result in data_point['api'].items():
                if endpoint not in endpoint_stats:
                    endpoint_stats[endpoint] = []
                if result['success'] and result['response_time_ms']:
                    endpoint_stats[endpoint].append(result['response_time_ms'])
        
        for endpoint, times in endpoint_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"  {endpoint}:")
                print(f"    Average: {avg_time:.0f}ms | Min: {min_time:.0f}ms | Max: {max_time:.0f}ms")
        
        # Generate recommendations
        latest_system = data_points[-1]['system']
        latest_api = data_points[-1]['api']
        recommendations = self.get_recommendations(latest_system, latest_api)
        
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                icon = "⚠️" if rec['type'] == 'warning' else "❌" if rec['type'] == 'error' else "💡"
                print(f"  {icon} {rec['message']}")
        
        print("\n✅ Monitoring complete!")

def main():
    parser = argparse.ArgumentParser(description='Monitor Contact Center AI performance')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Monitoring duration in minutes (default: 5)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Data collection interval in seconds (default: 30)')
    parser.add_argument('--api-url', default='http://localhost:8000',
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--single', action='store_true',
                       help='Run single check instead of continuous monitoring')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(api_url=args.api_url)
    
    if args.single:
        print("🔍 Running single performance check...")
        system_info = monitor.get_system_info()
        api_results = monitor.test_api_performance()
        recommendations = monitor.get_recommendations(system_info, api_results)
        
        print(f"\n📊 System Status:")
        print(f"  CPU: {system_info['cpu_percent']:.1f}%")
        print(f"  Memory: {system_info['memory_percent']:.1f}% ({system_info['memory_used_gb']:.1f}GB used)")
        print(f"  Available Memory: {system_info['memory_available_gb']:.1f}GB")
        
        print(f"\n🌐 API Response Times:")
        for endpoint, result in api_results.items():
            if result['success']:
                print(f"  {endpoint}: {result['response_time_ms']:.0f}ms")
            else:
                print(f"  {endpoint}: Failed - {result.get('error', 'Unknown error')}")
        
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                icon = "⚠️" if rec['type'] == 'warning' else "❌" if rec['type'] == 'error' else "💡"
                print(f"  {icon} {rec['message']}")
    else:
        monitor.run_monitoring(duration_minutes=args.duration, 
                             interval_seconds=args.interval)

if __name__ == "__main__":
    main()