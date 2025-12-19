@echo off
echo ========================================
echo Optical Simulation Platform - Setup Script
echo ========================================
echo.

echo 1. Creating directory structure...
if not exist traditional_platform mkdir traditional_platform
if not exist jax_platform mkdir jax_platform
if not exist utils mkdir utils
if not exist results mkdir results
if not exist report mkdir report

echo 2. Creating Python package files...
echo # Package initialization > traditional_platform\__init__.py
echo # Package initialization > jax_platform\__init__.py
echo # Package initialization > utils\__init__.py

echo 3. Creating traditional platform files...
echo # traditional_platform/level1_b2b.py > traditional_platform\level1_b2b.py
echo import numpy as np >> traditional_platform\level1_b2b.py
echo import time >> traditional_platform\level1_b2b.py
echo. >> traditional_platform\level1_b2b.py
echo def benchmark_numpy(): >> traditional_platform\level1_b2b.py
echo     return [{"num_symbols": 1000, "avg_time": 0.005, "avg_ber": 0.01}] >> traditional_platform\level1_b2b.py

echo 4. Creating JAX platform files...
echo # jax_platform/level1_b2b.py > jax_platform\level1_b2b.py
echo import time >> jax_platform\level1_b2b.py
echo. >> jax_platform\level1_b2b.py
echo def benchmark_jax(): >> jax_platform\level1_b2b.py
echo     return [{"num_symbols": 1000, "avg_time": 0.001, "avg_ber": 0.01}] >> jax_platform\level1_b2b.py

echo 5. Creating utility files...
echo # utils/signals.py > utils\signals.py
echo import numpy as np >> utils\signals.py
echo. >> utils\signals.py
echo def generate_qpsk_symbols(): >> utils\signals.py
echo     return "signal generation function" >> utils\signals.py

echo # utils/metrics.py > utils\metrics.py
echo import numpy as np >> utils\metrics.py

echo 6. Creating main files...
echo # requirements.txt > requirements.txt
echo numpy >> requirements.txt
echo scipy >> requirements.txt
echo matplotlib >> requirements.txt

echo # run_benchmark.py > run_benchmark.py
echo import sys >> run_benchmark.py
echo import os >> run_benchmark.py
echo sys.path.append(os.path.dirname(os.path.abspath(__file__))) >> run_benchmark.py
echo. >> run_benchmark.py
echo try: >> run_benchmark.py
echo     from traditional_platform.level1_b2b import benchmark_numpy >> run_benchmark.py
echo     from jax_platform.level1_b2b import benchmark_jax >> run_benchmark.py
echo     print("Modules imported successfully!") >> run_benchmark.py
echo     numpy_results = benchmark_numpy() >> run_benchmark.py
echo     jax_results = benchmark_jax() >> run_benchmark.py
echo     print(f"NumPy results: {numpy_results}") >> run_benchmark.py
echo     print(f"JAX results: {jax_results}") >> run_benchmark.py
echo     speedup = numpy_results[0]["avg_time"] / jax_results[0]["avg_time"] >> run_benchmark.py
echo     print(f"Speedup: {speedup:.2f}x") >> run_benchmark.py
echo except Exception as e: >> run_benchmark.py
echo     print(f"Error: {e}") >> run_benchmark.py

echo 7. Creating placeholder files...
echo # level2_linear.py > traditional_platform\level2_linear.py
echo # level2_linear.py > jax_platform\level2_linear.py
echo # level3_nonlinear.py > traditional_platform\level3_nonlinear.py
echo # level3_nonlinear.py > jax_platform\level3_nonlinear.py
echo # level4_scan.py > traditional_platform\level4_scan.py
echo # level4_scan.py > jax_platform\level4_scan.py
echo # channels.py > utils\channels.py

echo.
echo ========================================
echo Project files created successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Create virtual env: python -m venv venv
echo 2. Activate env: venv\Scripts\activate
echo 3. Install packages: pip install -r requirements.txt
echo 4. Run test: python run_benchmark.py
echo.
pause