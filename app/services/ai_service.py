import json
import logging
import os
import re
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import settings
from app.core.database import VideoType
from app.schemas.video import ChapterPoint, VideoAnalysisResult

logger = logging.getLogger(__name__)

class AIService:
    """AI服务，负责与大语言模型API交互，使用LangChain框架"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL
        self.model_name = settings.OPENAI_MODEL_NAME
        self.provider = settings.LLM_PROVIDER
        
        # 初始化LangChain模型
        self._initialize_model()
    
    def _initialize_model(self):
        """根据配置初始化适当的LangChain模型"""
        try:
            # 添加调试日志，记录实际使用的配置值
            logger.info(f"初始化模型配置: provider={self.provider}, model={self.model_name}, base_url={self.base_url}")
            
            if self.provider == "openai":
                # 为OpenAI提供一个配置，避免proxies参数问题
                from langchain_openai.chat_models.base import ChatOpenAI as ConfigurableChatOpenAI
                
                self.chat_model = ConfigurableChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=0.7,
                    max_tokens=2000,
                    # 添加http_client=None解决proxies问题
                    http_client=None
                )
            elif self.provider == "azure":
                # Azure OpenAI实现
                from langchain_openai import AzureChatOpenAI
                
                self.chat_model = AzureChatOpenAI(
                    deployment_name=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
                    temperature=0.7,
                    max_tokens=2000,
                )
            elif self.provider == "dashscope":
                # 阿里云通义千问 - 使用OpenAI兼容接口调用
                from openai import OpenAI
                
                # 记录DashScope配置信息
                logger.info(f"使用DashScope兼容接口: api_key={self.api_key[:5]}***, base_url={self.base_url}")
                
                # 创建一个OpenAI兼容的客户端
                openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    # 移除可能自动添加的proxies参数
                    http_client=None
                )
                
                # 创建一个兼容LangChain接口的包装类
                class DashScopeOpenAIWrapper:
                    def __init__(self, client, model_name, temperature=0.7, max_tokens=2000):
                        self.client = client
                        self.model_name = model_name
                        self.temperature = temperature
                        self.max_tokens = max_tokens
                
                    def __call__(self, messages):
                        # 将LangChain消息格式转换为OpenAI格式
                        openai_messages = []
                        for message in messages:
                            if isinstance(message, HumanMessage):
                                openai_messages.append({"role": "user", "content": message.content})
                            elif isinstance(message, SystemMessage):
                                openai_messages.append({"role": "system", "content": message.content})
                            elif isinstance(message, AIMessage):
                                openai_messages.append({"role": "assistant", "content": message.content})
                            else:
                                # 处理其他类型的消息
                                role = message.type if hasattr(message, 'type') else "user"
                                openai_messages.append({"role": role, "content": message.content})
                        
                        try:
                            # 调用OpenAI兼容接口
                            logger.info(f"调用DashScope API: model={self.model_name}, messages_count={len(openai_messages)}")
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=openai_messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                            )
                            
                            # 创建一个类，模拟LangChain响应对象
                            class LangChainResponse:
                                def __init__(self, content):
                                    self.content = content
                            
                            # 从响应中提取内容并返回LangChain格式响应
                            content = response.choices[0].message.content
                            return LangChainResponse(content)
                        except Exception as e:
                            error_msg = f"调用通义千问API失败: {str(e)}"
                            logger.error(error_msg)
                            raise Exception(error_msg)
                
                self.chat_model = DashScopeOpenAIWrapper(
                    client=openai_client,
                    model_name=self.model_name,
                    temperature=0.7,
                    max_tokens=2000,
                )
            else:
                # 默认使用OpenAI实现
                logger.warning(f"未知的模型提供商: {self.provider}，使用OpenAI作为默认提供商")
                # 为OpenAI提供一个配置，避免proxies参数问题
                from langchain_openai.chat_models.base import ChatOpenAI as ConfigurableChatOpenAI
                
                self.chat_model = ConfigurableChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=0.7,
                    max_tokens=2000,
                    # 添加http_client=None解决proxies问题
                    http_client=None
                )
        except Exception as e:
            logger.error(f"初始化LangChain模型失败: {str(e)}")
            raise
    
    def _call_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用大模型API，使用LangChain实现"""
        try:
            # 将消息格式转换为LangChain格式
            langchain_messages = []
            for message in messages:
                if message["role"] == "system":
                    langchain_messages.append(SystemMessage(content=message["content"]))
                elif message["role"] == "user":
                    langchain_messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=message["content"]))
                # 可以根据需要添加其他角色的处理
            
            # 调用模型
            response = self.chat_model(langchain_messages)
            
            # 检查response类型并提取content
            content = ""
            if hasattr(response, "content"):
                # 标准LangChain响应对象
                content = response.content
            elif isinstance(response, dict) and "content" in response:
                # 字典类型响应
                content = response["content"]
            else:
                # 未知格式响应，尝试转换为字符串
                logger.warning(f"未知响应格式: {type(response)}, 尝试转换为字符串")
                content = str(response)
            
            # 将LangChain响应格式转换回原来的格式
            return {
                "choices": [
                    {
                        "message": {
                            "content": content
                        }
                    }
                ]
            }
        except Exception as e:
            logger.error(f"调用大模型API失败: {str(e)}")
            raise
    
    def _split_transcript(self, transcript: str, max_chunk_size: int = None) -> List[str]:
        """
        将长文本转录切分为适合模型处理的片段
        
        Args:
            transcript: 完整转录文本
            max_chunk_size: 每个片段的最大字符数（近似tokens数），为None时使用配置值
            
        Returns:
            List[str]: 文本片段列表
        """
        # 使用配置中的分片大小或默认值
        if max_chunk_size is None:
            max_chunk_size = settings.TRANSCRIPT_CHUNK_SIZE
        
        # 如果转录文本较短，直接返回
        if len(transcript) <= max_chunk_size:
            return [transcript]
        
        # 按句子切分
        sentences = re.split(r'(。|！|？|\.|\!|\?)', transcript)
        chunks = []
        current_chunk = ""
        
        # 每两个元素组合成一个完整的句子（句子内容+标点符号）
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + sentences[i+1] if i+1 < len(sentences) else sentences[i]
            
            # 如果当前块加上新句子会超出大小限制，则存储当前块并创建新块
            if len(current_chunk) + len(sentence) > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    def _analyze_transcript_chunk(self, chunk: str, video_type: VideoType, chunk_index: int, total_chunks: int, duration: float) -> Dict[str, Any]:
        """
        分析单个转录文本块
        
        Args:
            chunk: 转录文本块
            video_type: 视频类型
            chunk_index: 当前块索引
            total_chunks: 总块数
            duration: 视频总时长
            
        Returns:
            Dict: 分析结果（包含部分章节点和部分解说）
        """
        # 构建提示
        video_type_text = "电影" if video_type == VideoType.MOVIE else "教学视频"
        
        context = f"""你是一位专业的视频内容分析师。我会给你一段视频的文字记录，这是{video_type_text}内容的第{chunk_index+1}/{total_chunks}部分。
视频总时长为{duration}秒。请仔细分析这部分内容。"""
        
        task = ""
        if video_type == VideoType.MOVIE:
            task = """请分析这部分电影内容，完成以下任务：
1. 识别这部分中的关键情节点和重要时刻
2. 为这部分内容创建解说脚本，包括情节解析、角色分析和电影赏析要点
            
请以JSON格式返回，包含以下字段：
{
    "chapters": [
        {"time": 时间点(秒), "title": "章节标题"}
    ],
    "narration_script": "这部分的解说脚本"
}

注意：
- 时间点是相对于整个视频的估计时间
- 解说脚本应该专业、有深度，包括情节分析、角色心理和电影技巧分析"""
        else:
            task = """请分析这部分教学视频内容，完成以下任务：
1. 识别这部分中的知识点和章节分隔点
2. 为这部分内容创建解说脚本，总结知识要点，并提供更清晰的讲解

请以JSON格式返回，包含以下字段：
{
    "chapters": [
        {"time": 时间点(秒), "title": "章节标题"}
    ],
    "narration_script": "这部分的解说脚本"
}

注意：
- 时间点是相对于整个视频的估计时间
- 解说脚本应该清晰、结构化，帮助学习者更好地理解知识点"""
        
        # 构建消息
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": f"这是视频转录文本的第{chunk_index+1}/{total_chunks}部分:\n\n{chunk}\n\n{task}"}
        ]
        
        # 调用模型
        response = self._call_model(messages)
        content = response["choices"][0]["message"]["content"]
        
        # 解析结果
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"模型返回的不是有效JSON，尝试提取有用信息，原始响应: {content[:200]}...")
            # 如果不是JSON格式，返回空结果
            return {
                "chapters": [],
                "narration_script": content
            }
        
    def _merge_analysis_results(self, results: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """
        合并多个分析结果
        
        Args:
            results: 多个分析结果
            duration: 视频总时长
            
        Returns:
            Dict: 合并后的分析结果
        """
        merged_chapters = []
        merged_narration = ""
        
        # 收集所有章节点
        for result in results:
            if "chapters" in result:
                merged_chapters.extend(result["chapters"])
            if "narration_script" in result:
                merged_narration += result["narration_script"] + "\n\n"
        
        # 章节点去重和排序
        unique_chapters = {}
        for chapter in merged_chapters:
            # 使用时间点的近似值作为键以去除相似时间点
            time_key = round(float(chapter.get("time", 0)) / 10) * 10  # 舍入到最接近的10秒
            # 保留最详细的章节标题
            if time_key not in unique_chapters or len(chapter.get("title", "")) > len(unique_chapters[time_key].get("title", "")):
                unique_chapters[time_key] = chapter
        
        # 将去重后的章节点转换回列表并按时间排序
        sorted_chapters = sorted(unique_chapters.values(), key=lambda x: float(x.get("time", 0)))
        
        # 确保章节点时间不超过视频总时长
        final_chapters = [ch for ch in sorted_chapters if float(ch.get("time", 0)) <= duration]
        
        # 如果没有识别出章节点，添加默认章节点
        if not final_chapters:
            # 添加开始、四分之一、中间、四分之三和结束的时间点
            for i in range(5):
                time_point = (duration * i) / 4
                title = "开场" if i == 0 else "结尾" if i == 4 else f"第{i}部分"
                final_chapters.append({"time": time_point, "title": title})
        
        # 为解说脚本添加总结
        final_narration = merged_narration.strip()
        if len(results) > 1:  # 只有多个结果时才添加总结
            # 调用模型生成总结
            summary_messages = [
                {"role": "system", "content": "你是一位视频内容的总结专家。请对以下解说脚本进行总结，提炼出最重要的内容，保持内容的连贯性和逻辑性。"},
                {"role": "user", "content": f"以下是一段解说脚本，请帮我总结出一个简短的开头和结尾部分，使整体内容更加完整：\n\n{final_narration}"}
            ]
            
            try:
                summary_response = self._call_model(summary_messages)
                summary = summary_response["choices"][0]["message"]["content"]
                
                # 将总结添加到解说脚本中
                final_narration = f"{summary}\n\n{final_narration}"
            except Exception as e:
                logger.error(f"生成解说总结失败: {str(e)}")
                # 如果生成总结失败，使用原始解说
        
        return {
            "chapters": final_chapters,
            "narration_script": final_narration
        }

    def analyze_transcript(self, transcript: str, video_type: VideoType, duration: float) -> VideoAnalysisResult:
        """分析视频文字记录，生成章节点和解说脚本"""
        
        # 处理空转录的情况
        if not transcript or transcript.strip() == "":
            logger.warning("转录文本为空，生成基本章节结构")
            # 创建基本章节结构
            chapter_points = []
            # 每15分钟一个章节
            interval = 15 * 60  # 15分钟（秒）
            for i in range(0, int(duration), interval):
                title = "开场" if i == 0 else f"第{i//interval+1}部分"
                chapter_points.append(ChapterPoint(time=float(i), title=title))
            
            # 确保有结尾章节
            if duration > 60 and (duration - chapter_points[-1].time) > 60:
                chapter_points.append(ChapterPoint(time=duration-30, title="结尾"))
            
            # 生成基本解说
            basic_narration = f"这是一段时长为{int(duration//60)}分{int(duration%60)}秒的{video_type.name}视频。由于无法获取有效的语音转录，无法提供详细解说。"
            
            return VideoAnalysisResult(
                video_type=video_type,
                duration=duration,
                chapter_points=chapter_points,
                transcript=transcript,
                narration_script=basic_narration
            )
            
        try:
            # 将长转录文本分割成较小的块
            chunks = self._split_transcript(transcript)
            logger.info(f"将转录文本分成了{len(chunks)}个块进行处理")
            
            # 如果只有一个块，直接处理
            if len(chunks) == 1:
                return self._process_single_transcript(transcript, video_type, duration)
            
            # 对每个块进行处理
            results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"处理第{i+1}/{len(chunks)}个转录块")
                result = self._analyze_transcript_chunk(chunk, video_type, i, len(chunks), duration)
                results.append(result)
            
            # 合并结果
            merged_result = self._merge_analysis_results(results, duration)
            
            # 解析章节点
            chapter_points = []
            if "chapters" in merged_result:
                for chapter in merged_result["chapters"]:
                    chapter_points.append(
                        ChapterPoint(
                            time=float(chapter.get("time", 0)), 
                            title=chapter.get("title", "")
                        )
                    )
            
            # 获取解说脚本
            narration_script = merged_result.get("narration_script", "")
            
            # 确保章节点按时间排序
            chapter_points.sort(key=lambda x: x.time)
            
            return VideoAnalysisResult(
                video_type=video_type,
                duration=duration,
                chapter_points=chapter_points,
                transcript=transcript,
                narration_script=narration_script
            )
            
        except Exception as e:
            logger.error(f"分析视频文字记录失败: {str(e)}")
            raise
            
    def _process_single_transcript(self, transcript: str, video_type: VideoType, duration: float) -> VideoAnalysisResult:
        """处理单个完整的转录文本（原有方法的逻辑）"""
        
        # 构建提示
        video_type_text = "电影" if video_type == VideoType.MOVIE else "教学视频"
        
        context = f"""你是一位专业的视频内容分析师。我会给你一段视频的文字记录，这是{video_type_text}内容。
视频总时长为{duration}秒。请仔细分析内容。"""
        
        task = ""
        if video_type == VideoType.MOVIE:
            task = """请分析这部电影内容，完成以下任务：
1. 识别关键情节点和重要时刻，创建章节结构
2. 创建专业的解说脚本，包括情节解析、角色分析和电影赏析要点
            
请以JSON格式返回，包含以下字段：
{
    "chapters": [
        {"time": 时间点(秒), "title": "章节标题"}
    ],
    "narration_script": "解说脚本"
}

注意：
- 时间点是相对于整个视频的估计时间，从0秒到视频结束
- 解说脚本应该专业、有深度，包括情节分析、角色心理和电影技巧分析"""
        else:
            task = """请分析这段教学视频内容，完成以下任务：
1. 识别知识点和章节分隔点，创建章节结构
2. 创建解说脚本，总结知识要点，并提供更清晰的讲解

请以JSON格式返回，包含以下字段：
{
    "chapters": [
        {"time": 时间点(秒), "title": "章节标题"}
    ],
    "narration_script": "解说脚本"
}

注意：
- 时间点是相对于整个视频的估计时间，从0秒到视频结束
- 解说脚本应该清晰、结构化，帮助学习者更好地理解知识点"""
        
        # 构建消息
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": f"视频转录文本:\n\n{transcript}\n\n{task}"}
        ]
        
        # 调用模型
        try:
            response = self._call_model(messages)
            content = response["choices"][0]["message"]["content"]
            
            # 尝试解析JSON
            try:
                result_data = json.loads(content)
                
                # 解析章节点
                chapter_points = []
                if "chapters" in result_data:
                    for chapter in result_data["chapters"]:
                        chapter_points.append(
                            ChapterPoint(
                                time=float(chapter.get("time", 0)), 
                                title=chapter.get("title", "")
                            )
                        )
                
                # 获取解说脚本
                narration_script = result_data.get("narration_script", "")
                
                # 确保章节点按时间排序
                chapter_points.sort(key=lambda x: x.time)
                
                return VideoAnalysisResult(
                    video_type=video_type,
                    duration=duration,
                    chapter_points=chapter_points,
                    transcript=transcript,
                    narration_script=narration_script
                )
            except json.JSONDecodeError:
                # 如果不是JSON格式，尝试提取有用信息
                logger.warning("模型返回的不是有效JSON，尝试提取有用信息")
                return VideoAnalysisResult(
                    video_type=video_type,
                    duration=duration,
                    chapter_points=[],
                    transcript=transcript,
                    narration_script=content
                )
        except Exception as e:
            logger.error(f"分析视频文字记录失败: {str(e)}")
            raise

# 创建AI服务单例
ai_service = AIService() 