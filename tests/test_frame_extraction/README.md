# 关键帧提取器测试

本目录包含针对不同关键帧提取策略的测试用例。

## 测试文件结构

- `base_test.py`: 测试基类，包含通用设置和验证方法
- `test_ffmpeg.py`: FFmpeg提取器测试
- `test_moviepy.py`: MoviePy提取器测试
- `test_opencv.py`: OpenCV提取器测试
- `test_scene_detect.py`: 场景检测提取器测试
- `test_factory.py`: 工厂方法测试
- `run_all_tests.py`: 运行所有测试的脚本

## 运行测试

### 运行所有测试

```bash
python tests/test_frame_extraction/run_all_tests.py
```

### 运行单个提取器测试

```bash
# FFmpeg提取器测试
python tests/test_frame_extraction/test_ffmpeg.py

# MoviePy提取器测试
python tests/test_frame_extraction/test_moviepy.py

# OpenCV提取器测试
python tests/test_frame_extraction/test_opencv.py

# 场景检测提取器测试
python tests/test_frame_extraction/test_scene_detect.py

# 工厂方法测试
python tests/test_frame_extraction/test_factory.py
```

## 注意事项

1. 测试依赖于示例视频文件的存在，路径为: `/home/yxhpy/project/znjjrj/uploads/28927593393-1-192.mp4`
2. 如需使用其他视频文件进行测试，请修改 `base_test.py` 中的 `test_video_path` 变量 