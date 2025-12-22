"""
Quick test script to verify model can be loaded and make predictions.

This is a minimal test that doesn't require the full dataset.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Quick Model Test")
print("=" * 60)
print()

try:
    print("Step 1: Importing modules...")
    from src.model import BannerClassifier, ModelConfig
    print("[OK] Modules imported successfully")
    print()

    print("Step 2: Creating model configuration...")
    config = ModelConfig(
        use_quantization=True,
        quantization_bits=4,
        use_few_shot=True
    )
    print(f"[OK] Config created: {config.model_name}")
    print(f"  - Quantization: {config.use_quantization} ({config.quantization_bits}-bit)")
    print(f"  - Few-shot: {config.use_few_shot}")
    print()

    print("Step 3: Initializing classifier...")
    classifier = BannerClassifier(config)
    print("[OK] Classifier initialized")
    print()

    print("Step 4: Loading model (this may take 1-2 minutes first time)...")
    print("  Note: Model will be downloaded from HuggingFace if not cached")
    start_time = time.time()
    classifier.load_model()
    load_time = time.time() - start_time
    print(f"[OK] Model loaded in {load_time:.2f} seconds")
    print()

    print("Step 5: Testing predictions...")
    test_cases = [
        ("SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5", "ssh_server"),
        ("HTTP/1.1 200 OK\r\nServer: nginx/1.18.0", "web_server"),
        ("220 mail.example.com ESMTP Postfix", "mail_server"),
        ("MySQL 8.0.33", "database"),
        ("220 ProFTPD 1.3.6 Server", "ftp_server"),
    ]

    for banner, expected in test_cases:
        start = time.time()
        result = classifier.predict(banner)
        latency = time.time() - start
        
        status = "[OK]" if result['category'] == expected else "[WARN]"
        print(f"{status} Banner: {banner[:50]}...")
        print(f"  Predicted: {result['category']} (expected: {expected})")
        print(f"  Latency: {latency:.3f}s")
        print()

    print("Step 6: Model information...")
    info = classifier.get_model_info()
    print(f"  Model: {info['model_name']}")
    print(f"  Device: {info['device']}")
    print(f"  Quantization: {info['quantization']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print()

    print("=" * 60)
    print("[SUCCESS] Model test completed successfully!")
    print("=" * 60)

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("  Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

