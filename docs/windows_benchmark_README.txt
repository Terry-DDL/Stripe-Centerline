Stripe Centerline Windows EXE vs Python A/B Benchmark

准备：
1. 安装 python.org 的 64-bit Python 3.11.9（勾选 Python Launcher）。
2. 完整解压 StripeCenterline-Windows-Benchmark artifact。
3. 不要移动或删除 _internal、source 或其他文件。

运行：
1. 双击 run_windows_ab_benchmark.bat。
2. 首次运行会自动建立隔离环境并安装锁定版本的 NumPy/OpenCV。
3. 脚本会依次执行 EXE 和 Python 源码，各 1 次 warm-up + 10 次 measured。
4. 两边都使用固定图片、固定点 (775,427)、固定 ROI 和相同计时定义。
5. 完成后结果位于 windows_ab_results 文件夹。

请发回整个 windows_ab_results 文件夹，至少应包含：
- frozen_exe/benchmark_results.json
- python_source/benchmark_results.json
- ab_summary.json
- ab_summary.txt

如失败，请同时提供命令窗口截图。无需选择图片、点击参考点或输入参数。
