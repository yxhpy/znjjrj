#!/usr/bin/env python
"""
运行所有帧提取器测试
"""
import unittest
import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置路径
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(current_dir))

logger.info(f"当前测试目录: {current_dir}")
logger.info(f"项目根目录: {base_dir}")
logger.info(f"Python路径: {sys.path}")

if __name__ == '__main__':
    # 确保输出目录存在
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建测试输出目录: {output_dir}")
    
    # 创建测试加载器
    loader = unittest.TestLoader()
    
    # 找到当前目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载指定目录的所有测试
    logger.info(f"开始加载测试套件，从目录: {test_dir}")
    test_suite = loader.discover(test_dir, pattern='test_*.py')
    
    # 创建测试运行器
    logger.info("开始运行测试...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    logger.info(f"测试完成. 运行: {result.testsRun}, 错误: {len(result.errors)}, 失败: {len(result.failures)}, 跳过: {len(result.skipped)}") 