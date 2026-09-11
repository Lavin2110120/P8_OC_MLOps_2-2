import subprocess
import os

test_files = [
    "tests/test_api.py",
    "tests/test_load.py",
    "tests/test_integration_db.py",
    "tests/test_security_resilience.py",
    "tests/test_monitoring_observability.py",
]

for test_file in test_files:
    output_dir = f"coverage_reports/{os.path.splitext(os.path.basename(test_file))[0]}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "pytest",
        "--cov=src",
        f"--cov-report=html:{output_dir}",
        "--cov-config=.coveragerc",
        test_file,
    ]
    subprocess.run(cmd, check=True)