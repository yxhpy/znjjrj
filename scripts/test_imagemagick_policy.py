#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ImageMagick安全策略测试脚本

用于检查ImageMagick的安全策略配置，特别是@符号文件操作的限制
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_imagemagick_version():
    """检查ImageMagick版本"""
    logger.info("检查ImageMagick版本...")
    
    try:
        # 执行identify -version命令
        result = subprocess.run(
            ['identify', '-version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        version_info = result.stdout
        logger.info(f"ImageMagick版本信息:\n{version_info}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"执行identify命令失败: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ImageMagick未安装或未在PATH中")
        return False

def check_policy_file():
    """检查ImageMagick策略文件"""
    logger.info("检查ImageMagick策略文件...")
    
    # 常见的策略文件位置
    policy_paths = [
        '/etc/ImageMagick-6/policy.xml',
        '/etc/ImageMagick/policy.xml',
        '/usr/local/etc/ImageMagick-6/policy.xml',
        '/usr/local/etc/ImageMagick/policy.xml'
    ]
    
    found = False
    
    for path in policy_paths:
        if os.path.exists(path):
            logger.info(f"找到策略文件: {path}")
            found = True
            
            # 读取文件内容
            try:
                with open(path, 'r') as f:
                    content = f.read()
                
                # 检查@*限制
                if 'domain="path" rights="none" pattern="@*"' in content:
                    logger.warning(f"检测到@*文件访问限制")
                else:
                    logger.info(f"未检测到@*文件访问限制")
                
                # 记录整个策略文件的内容
                logger.debug(f"策略文件内容:\n{content}")
            except Exception as e:
                logger.error(f"无法读取策略文件: {str(e)}")
    
    if not found:
        logger.warning("未找到ImageMagick策略文件")
    
    return found

def test_at_file_operation():
    """测试@文件操作"""
    logger.info("测试@文件操作...")
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        # 创建一个临时文本文件
        f.write(b"Hello, ImageMagick!")
        temp_file_path = f.name
    
    try:
        # 创建使用@参数的命令
        at_path = f"@{temp_file_path}"
        
        # 尝试使用convert命令绘制文本
        cmd = ['convert', '-size', '200x100', 'xc:white', '-annotate', '+10+50', at_path, 'test_output.png']
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("@文件操作成功")
            
            # 检查输出文件
            if os.path.exists('test_output.png'):
                logger.info("成功创建输出文件")
                os.remove('test_output.png')
            else:
                logger.warning("命令执行成功但未创建输出文件")
            
            return True
        else:
            logger.error(f"@文件操作失败: {result.stderr}")
            
            # 检查安全策略错误
            if "attempt to perform an operation not allowed by the security policy" in result.stderr:
                logger.error("被安全策略阻止")
            
            return False
    except Exception as e:
        logger.error(f"测试@文件操作时出错: {str(e)}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def suggest_policy_fix():
    """建议修复策略文件的方法"""
    logger.info("\n修复安全策略的建议:")
    logger.info("1. 备份当前策略文件:")
    logger.info("   sudo cp /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.backup")
    logger.info("")
    logger.info("2. 方法一: 注释掉@*限制行:")
    logger.info("   找到包含'domain=\"path\" rights=\"none\" pattern=\"@*\"'的行，")
    logger.info("   在其前后添加<!-- -->注释标记")
    logger.info("")
    logger.info("3. 方法二: 创建新的简化策略文件:")
    logger.info("""   sudo tee /etc/ImageMagick-6/policy.xml > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE policymap [
  <!ELEMENT policymap (policy)+>
  <!ATTLIST policymap xmlns CDATA #FIXED ''>
  <!ELEMENT policy EMPTY>
  <!ATTLIST policy xmlns CDATA #FIXED '' domain NMTOKEN #REQUIRED
    name NMTOKEN #IMPLIED pattern CDATA #IMPLIED rights NMTOKEN #IMPLIED
    stealth NMTOKEN #IMPLIED value CDATA #IMPLIED>
]>
<policymap>
  <policy domain="resource" name="memory" value="256MiB"/>
  <policy domain="resource" name="width" value="32KP"/>
  <policy domain="resource" name="height" value="32KP"/>
  <policy domain="resource" name="map" value="512MiB"/>
  <policy domain="resource" name="disk" value="1GiB"/>
  <policy domain="resource" name="area" value="1GB"/>
  <policy domain="resource" name="time" value="120"/>
</policymap>
EOF""")

def main():
    """主函数"""
    logger.info("开始ImageMagick安全策略测试...")
    
    success = True
    
    # 检查ImageMagick版本
    if not check_imagemagick_version():
        logger.error("未检测到ImageMagick，请先安装")
        return
    
    # 检查策略文件
    check_policy_file()
    
    # 测试@文件操作
    if test_at_file_operation():
        logger.info("@文件操作测试通过")
    else:
        logger.error("@文件操作测试失败")
        success = False
    
    # 如果测试失败，提供修复建议
    if not success:
        suggest_policy_fix()
    
    logger.info("ImageMagick安全策略测试完成")

if __name__ == "__main__":
    main() 