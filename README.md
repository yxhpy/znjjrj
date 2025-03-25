# 智能视频解说系统

一个基于AI的视频自动剪辑和解说系统，支持视频分析、文本生成和语音合成功能。

## 功能特点

- 视频自动分析和场景识别
- 智能解说文本生成
- 多种语音合成引擎支持
- 自动字幕和解说添加

## TTS语音合成服务

系统支持多种语音合成(TTS)引擎，可通过.env文件进行配置：

### 配置说明

在.env文件中配置以下参数：

```
# TTS服务配置
TTS_SERVICE_TYPE=gtts  # 可选: gtts, coqui, chattts

# gTTS配置 (Google TTS)
GTTS_LANG=zh-cn
GTTS_SLOW=False

# Coqui TTS配置 (高质量开源TTS)
COQUI_MODEL_NAME=tts_models/zh-CN/baker/tacotron2-DDC-GST
COQUI_VOCODER_NAME=vocoder_models/universal/libri-tts/fullband-melgan
COQUI_USE_GPU=False

# ChatTTS配置 (第三方API服务)
CHATTTS_API_KEY=your_chattts_api_key
CHATTTS_VOICE=default
CHATTTS_RATE=1.0
CHATTTS_PITCH=1.0
```

### 安装依赖

使用不同的TTS引擎需要安装相应的依赖：

1. **gTTS**（默认）：
   ```
   pip install gtts
   ```

2. **Coqui TTS**（高质量开源TTS）：
   ```
   pip install TTS torch
   ```

3. **ChatTTS**（第三方API服务）：
   ```
   pip install requests
   ```

也可以一次性安装所有依赖：
```
pip install -r requirements.txt
```

### 选择合适的TTS服务

- **gTTS**: 简单易用，不需要额外资源，但语音质量一般
- **Coqui TTS**: 高质量开源TTS，支持多种语言和声音，但需要下载模型（约1-2GB）并占用更多计算资源
- **ChatTTS**: 基于云API的服务，语音质量高且无需本地计算资源，但需要API密钥和网络连接

### 使用示例

系统会根据配置自动选择相应的TTS服务，无需修改代码。

## 安装和运行

1. 克隆项目仓库
   ```
   git clone https://github.com/yourusername/video-narration-system.git
   cd video-narration-system
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

3. 配置.env文件
   ```
   cp .env.example .env
   # 编辑.env文件，设置相应参数
   ```

4. 运行系统
   ```
   uvicorn app.main:app --reload
   ```

5. 访问API文档
   ```
   http://localhost:8000/docs
   ```

## 设计模式

本项目使用了多种设计模式：

1. **策略模式**：用于实现不同的TTS服务策略
2. **工厂模式**：根据配置创建相应的服务实例
3. **单例模式**：确保服务只被初始化一次
4. **依赖注入**：在各组件间传递服务实例

## 许可证

MIT 