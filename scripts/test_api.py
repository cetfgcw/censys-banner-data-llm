"""
Simple script to test the API endpoints.

Usage:
    python scripts/test_api.py [--url http://localhost:8000]
"""

import argparse
import requests
import time
import json

def test_health(url):
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{url}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("  ✓ Health check passed")


def test_ready(url):
    """Test ready endpoint."""
    print("Testing /ready...")
    response = requests.get(f"{url}/ready", timeout=5)
    if response.status_code == 503:
        print("  ⚠ Service not ready (model may still be loading)")
        return False
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] == True
    print("  ✓ Readiness check passed")
    return True


def test_predict(url):
    """Test single prediction."""
    print("Testing /predict...")
    test_cases = [
        ("SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5", "ssh_server"),
        ("HTTP/1.1 200 OK\r\nServer: nginx/1.18.0", "web_server"),
        ("220 mail.example.com ESMTP Postfix", "mail_server"),
    ]
    
    for banner, expected_category in test_cases:
        response = requests.post(
            f"{url}/predict",
            json={"banner_text": banner},
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        print(f"  ✓ Predicted '{data['category']}' for banner (expected: {expected_category})")


def test_batch_predict(url):
    """Test batch prediction."""
    print("Testing /predict/batch...")
    banners = [
        "SSH-2.0-OpenSSH_8.2p1",
        "HTTP/1.1 200 OK",
        "220 mail.example.com ESMTP"
    ]
    
    response = requests.post(
        f"{url}/predict/batch",
        json={"banners": banners},
        timeout=60
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == len(banners)
    assert len(data["predictions"]) == len(banners)
    print(f"  ✓ Batch prediction: {data['total']} predictions in {data['processing_time']:.2f}s")


def test_metrics(url):
    """Test metrics endpoint."""
    print("Testing /metrics...")
    response = requests.get(f"{url}/metrics", timeout=5)
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    print("  ✓ Metrics endpoint working")


def test_info(url):
    """Test info endpoint."""
    print("Testing /info...")
    response = requests.get(f"{url}/info", timeout=5)
    if response.status_code == 503:
        print("  ⚠ Info not available (model not loaded)")
        return
    assert response.status_code == 200
    data = response.json()
    assert "model_info" in data
    print(f"  ✓ Model: {data['model_info']['model_name']}")
    print(f"  ✓ Device: {data['model_info']['device']}")


def main():
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--wait", type=int, default=0, help="Wait N seconds before testing (for model loading)")
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"Waiting {args.wait} seconds for model to load...")
        time.sleep(args.wait)
    
    print(f"\nTesting API at {args.url}\n")
    print("="*60)
    
    try:
        test_health(args.url)
        ready = test_ready(args.url)
        
        if not ready:
            print("\n⚠ Model not ready. Waiting 30 seconds and retrying...")
            time.sleep(30)
            ready = test_ready(args.url)
        
        if ready:
            test_predict(args.url)
            test_batch_predict(args.url)
            test_info(args.url)
        else:
            print("\n⚠ Skipping prediction tests - model not ready")
        
        test_metrics(args.url)
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to {args.url}")
        print("  Make sure the API is running: docker-compose up")
        return 1
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

