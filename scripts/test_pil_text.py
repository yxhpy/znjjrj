#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PIL文本渲染测试脚本

测试使用PIL库渲染文本，作为MoviePy TextClip的替代方案
"""

import os
import sys
import logging
from pathlib import Path
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import requests
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import TEMP_DIR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_fonts():
    """检查可用字体"""
    logger.info("检查可用字体...")
    
    # 常见中英文字体路径
    font_paths = [
        # DejaVu字体（支持Unicode）
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        # Ubuntu/Debian中文字体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",             
        "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
        # Ubuntu/Debian英文字体
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # 思源黑体/Noto字体（同时支持中英文）
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        # Windows常见字体
        "C:\\Windows\\Fonts\\msyh.ttc",                              # 微软雅黑
        "C:\\Windows\\Fonts\\simsun.ttc",                            # 宋体
        "C:\\Windows\\Fonts\\simhei.ttf",                            # 黑体
        "C:\\Windows\\Fonts\\arial.ttf",                             # Arial
        # 其他常见路径
        "/usr/share/fonts/TTF/SimHei.ttf",               
        "/usr/share/fonts/SimHei.ttf",
        "SimHei.ttf",                                                 # 当前目录
    ]
    
    # 系统字体目录，尝试查找更多字体
    system_font_dirs = [
        "/usr/share/fonts/",
        "/usr/local/share/fonts/",
        os.path.expanduser("~/.fonts/"),
        "C:\\Windows\\Fonts\\"
    ]
    
    found_fonts = []
    
    # 首先检查指定的字体路径
    for path in font_paths:
        if os.path.exists(path):
            logger.info(f"找到字体: {path}")
            found_fonts.append(path)
    
    # 如果没有找到任何字体，尝试在系统字体目录中查找
    if not found_fonts:
        logger.warning("未找到列表中的字体，尝试在系统字体目录中查找...")
        
        for font_dir in system_font_dirs:
            if os.path.exists(font_dir):
                # 常见字体文件扩展名
                for ext in ['.ttf', '.ttc', '.otf']:
                    # 在字体目录中递归查找字体文件
                    for root, dirs, files in os.walk(font_dir):
                        for file in files:
                            if file.lower().endswith(ext):
                                font_path = os.path.join(root, file)
                                logger.info(f"找到系统字体: {font_path}")
                                found_fonts.append(font_path)
                                # 找到一个就返回，避免搜索太多
                                if found_fonts:
                                    return found_fonts
    
    if not found_fonts:
        logger.warning("未找到任何可用中英文字体，将使用默认字体")
    
    return found_fonts

def test_pil_text_rendering(font_path=None):
    """测试PIL文本渲染"""
    logger.info("测试PIL文本渲染...")
    
    # 测试文本 - 使用Unicode字符串
    test_text = u"这是测试文本，包含中文和English"
    
    # 图像尺寸
    width, height = 800, 200
    
    # 创建图像
    img = Image.new('RGBA', (width, height), (0, 0, 0, 200))
    draw = ImageDraw.Draw(img)
    
    # 加载字体
    font = None
    if font_path and os.path.exists(font_path):
        try:
            # 加载字体并指定编码
            font = ImageFont.truetype(font_path, 32, encoding="utf-8")
            logger.info(f"使用字体: {font_path}")
        except Exception as e:
            logger.error(f"加载字体失败: {str(e)}")
    
    if font is None:
        # 使用默认字体
        font = ImageFont.load_default()
        logger.info("使用默认字体")
    
    # 绘制文本 - 确保是Unicode字符串
    draw.text((20, 20), u"这是测试文本，包含中文和English", fill=(255, 255, 255, 255), font=font)
    draw.text((20, 80), u"测试时间: " + str(uuid.uuid4()), fill=(255, 255, 255, 255), font=font)
    draw.text((20, 140), u"PIL文本渲染测试", fill=(255, 255, 255, 255), font=font)
    
    # 保存图像
    output_path = os.path.join(TEMP_DIR, f"pil_text_test_{uuid.uuid4()}.png")
    img.save(output_path)
    
    logger.info(f"已保存测试图像: {output_path}")
    return output_path

def test_multiple_text_lines():
    """测试多行文本渲染"""
    logger.info("测试多行文本渲染...")
    
    # 测试文本 - 使用Unicode字符串
    lines = [
        u"这是第一行测试文本",
        u"这是第二行测试文本，包含更多的内容",
        u"This is the third line with English text",
        u"第四行: 混合中英文 Mixed Chinese and English"
    ]
    
    # 图像尺寸
    width, height = 800, 300
    
    # 创建图像
    img = Image.new('RGBA', (width, height), (0, 0, 0, 200))
    draw = ImageDraw.Draw(img)
    
    # 检查可用字体
    fonts = check_fonts()
    if fonts:
        # 使用找到的第一个字体
        font = ImageFont.truetype(fonts[0], 24, encoding="utf-8")
        logger.info(f"多行文本使用字体: {fonts[0]}")
    else:
        # 使用默认字体（但可能无法显示中文）
        font = ImageFont.load_default()
        logger.warning("使用默认字体，可能无法正确显示中文")
    
    # 绘制多行文本
    line_height = 50
    for i, line in enumerate(lines):
        y_pos = 20 + i * line_height
        # 确保文本是Unicode字符串
        draw.text((20, y_pos), line, fill=(255, 255, 255, 255), font=font)
    
    # 保存图像
    output_path = os.path.join(TEMP_DIR, f"pil_multiline_test_{uuid.uuid4()}.png")
    img.save(output_path)
    
    logger.info(f"已保存多行文本测试图像: {output_path}")
    return output_path

def test_textbox_drawing():
    """测试文本框绘制"""
    logger.info("测试文本框绘制...")
    
    # 创建图像
    width, height = 800, 400
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))  # 透明背景
    draw = ImageDraw.Draw(img)
    
    # 绘制文本框
    margin = 20
    box_width = width - 2 * margin
    box_height = 80
    
    # 绘制多个文本框 - 使用Unicode字符串
    y_positions = [30, 150, 270]
    texts = [
        u"这是第一个文本框，有黑色背景和白色文字",
        u"This is the second text box with blue background",
        u"第三个文本框: 混合中英文 Mixed Chinese and English"
    ]
    
    bg_colors = [(0, 0, 0, 200), (0, 0, 128, 200), (128, 0, 0, 200)]
    text_colors = [(255, 255, 255, 255), (255, 255, 0, 255), (255, 255, 255, 255)]
    
    # 检查可用字体
    fonts = check_fonts()
    if fonts:
        # 使用找到的第一个字体，指定编码
        font = ImageFont.truetype(fonts[0], 24, encoding="utf-8")
        logger.info(f"文本框使用字体: {fonts[0]}")
    else:
        # 使用默认字体（但可能无法显示中文）
        font = ImageFont.load_default()
        logger.warning("使用默认字体，可能无法正确显示中文")
    
    for i, y_pos in enumerate(y_positions):
        # 绘制背景矩形
        box_coords = [(margin, y_pos), (margin + box_width, y_pos + box_height)]
        draw.rectangle(box_coords, fill=bg_colors[i])
        
        # 绘制文本
        text_margin = 10
        draw.text(
            (margin + text_margin, y_pos + text_margin), 
            texts[i], 
            fill=text_colors[i], 
            font=font
        )
    
    # 保存图像
    output_path = os.path.join(TEMP_DIR, f"pil_textbox_test_{uuid.uuid4()}.png")
    img.save(output_path)
    
    logger.info(f"已保存文本框测试图像: {output_path}")
    return output_path

def test_mixed_language():
    """测试中英文混合显示"""
    logger.info("测试中英文混合显示...")
    
    # 测试文本 - 使用Unicode字符串
    texts = [
        u"中文 English 混合文本测试",
        u"Testing Mixed 中英文 Characters",
        u"123456789 数字测试 Number Test",
        u"特殊符号 !@#$%^&*() Special Symbols"
    ]
    
    # 图像尺寸
    width, height = 800, 300
    
    # 创建图像
    img = Image.new('RGBA', (width, height), (0, 0, 0, 200))
    draw = ImageDraw.Draw(img)
    
    # 获取可用字体
    fonts = check_fonts()
    if not fonts:
        logger.error("没有找到可用字体，无法进行中英文混合显示测试")
        return None
    
    # 依次测试前三个字体（如果有）
    font_paths = fonts[:min(3, len(fonts))]
    y_position = 20
    
    for i, font_path in enumerate(font_paths):
        try:
            # 尝试加载字体，指定编码
            font = ImageFont.truetype(font_path, 24, encoding="utf-8")
            logger.info(f"测试字体 {i+1}: {font_path}")
            
            # 绘制文本
            for j, text in enumerate(texts):
                x_pos = 20
                y_pos = y_position + j * 30
                # 绘制字体名称标记
                if j == 0:
                    font_name = os.path.basename(font_path)
                    draw.text((x_pos, y_pos), u"Font: " + font_name, fill=(255, 255, 0, 255), font=font)
                    y_pos += 30
                
                # 确保文本是Unicode字符串
                draw.text((x_pos, y_pos), text, fill=(255, 255, 255, 255), font=font)
            
            # 调整下一个字体的起始位置
            y_position += 150
            
        except Exception as e:
            logger.error(f"使用字体 {font_path} 时出错: {str(e)}")
    
    # 保存图像
    output_path = os.path.join(TEMP_DIR, f"pil_mixed_language_test_{uuid.uuid4()}.png")
    img.save(output_path)
    
    logger.info(f"已保存中英文混合显示测试图像: {output_path}")
    return output_path

def download_font():
    """下载支持中英文的字体"""
    logger.info("尝试下载中英文字体...")
    
    # 可用的开源字体URL列表
    font_urls = [
        # 思源黑体 - Google Noto Sans CJK
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJK-Regular.otf",
        # 文泉驿微米黑
        "https://github.com/layerssss/wqy/raw/master/fonts/wqy-microhei.ttc",
        # DejaVu Sans
        "https://dejavu-fonts.github.io/Files/dejavu-sans-ttf-2.37.zip",
        # 落霞孤鹜字体
        "https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf",
        # 更纱黑体
        "https://github.com/be5invis/Sarasa-Gothic/releases/download/v0.40.3/sarasa-gothic-ttf-0.40.3.7z",
        # 霞鹜文楷
        "https://github.com/lxgw/LxgwWenKai-Lite/releases/download/v1.300/LXGWWenKaiLite-Regular.ttf"
    ]
    
    # 本地字体保存目录
    font_dir = os.path.join(TEMP_DIR, "fonts")
    os.makedirs(font_dir, exist_ok=True)
    
    # 尝试下载字体
    downloaded_font = None
    
    for url in font_urls:
        try:
            font_name = os.path.basename(url)
            local_font_path = os.path.join(font_dir, font_name)
            
            # 如果字体已存在，直接使用
            if os.path.exists(local_font_path):
                logger.info(f"本地已有字体: {local_font_path}")
                
                # 如果是压缩文件，则跳过
                if font_name.endswith(('.zip', '.7z')):
                    continue
                    
                downloaded_font = local_font_path
                break
            
            # 下载字体
            logger.info(f"正在下载字体: {url}")
            response = requests.get(url, stream=True, timeout=10)
            
            if response.status_code == 200:
                with open(local_font_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                logger.info(f"字体下载成功: {local_font_path}")
                
                # 如果是压缩文件，则跳过
                if font_name.endswith(('.zip', '.7z')):
                    continue
                    
                downloaded_font = local_font_path
                break
            else:
                logger.warning(f"下载失败: {url}, 状态码: {response.status_code}")
        
        except Exception as e:
            logger.error(f"下载字体时出错: {str(e)}")
    
    # 如果没有下载成功，尝试直接下载更简单的字体
    if downloaded_font is None:
        try:
            # 尝试下载Ubuntu字体
            simple_url = "https://github.com/ubuntu/fonts/raw/master/Ubuntu-R.ttf"
            font_name = "Ubuntu-R.ttf"
            local_font_path = os.path.join(font_dir, font_name)
            
            logger.info(f"尝试下载备用字体: {simple_url}")
            response = requests.get(simple_url, stream=True, timeout=10)
            
            if response.status_code == 200:
                with open(local_font_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                logger.info(f"备用字体下载成功: {local_font_path}")
                downloaded_font = local_font_path
        
        except Exception as e:
            logger.error(f"下载备用字体时出错: {str(e)}")
    
    return downloaded_font

def main():
    """主函数"""
    logger.info("开始PIL文本渲染测试...")
    
    # 首先尝试下载字体
    downloaded_font = download_font()
    
    # 检查可用字体
    fonts = check_fonts()
    
    # 如果有下载的字体，优先使用
    if downloaded_font:
        test_pil_text_rendering(downloaded_font)
        # 将下载的字体添加到字体列表的第一位
        if downloaded_font not in fonts:
            fonts.insert(0, downloaded_font)
    elif fonts:
        # 使用找到的第一个字体
        test_pil_text_rendering(fonts[0])
    else:
        # 使用默认字体
        test_pil_text_rendering()
    
    # 测试中英文混合显示
    test_mixed_language()
    
    # 测试多行文本
    test_multiple_text_lines()
    
    # 测试文本框
    test_textbox_drawing()
    
    logger.info("PIL文本渲染测试完成")

if __name__ == "__main__":
    main() 