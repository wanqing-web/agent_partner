import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class IntelligentMemoryManager:
    """
    基于语义的智能记忆管理器
    使用免费的本地 embedding 模型
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",  # 使用更小、更快下载的模型
                 similarity_threshold: float = 0.7,
                 importance_threshold: float = 0.5):
        """
        初始化智能记忆管理器

        Args:
            model_name: 使用的 embedding 模型名称（支持中文）
            similarity_threshold: 判断信息相似度的阈值
            importance_threshold: 判断信息重要性的阈值
        """
        # 禁用 SSL 验证
        ssl._create_default_https_context = ssl._create_unverified_context

        # 使用免费的本地 embedding 模型
        print(f"正在加载 embedding 模型: {model_name}...")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.use_embedding = True
            print("Embedding 模型加载完成！")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用纯规则匹配模式（不使用 embedding）")
            self.use_embedding = False
            self.embedding_model = None

        self.similarity_threshold = similarity_threshold
        self.importance_threshold = importance_threshold

        # 存储所有记忆
        self.memories: List[Dict[str, Any]] = []

        # 重要性模式（作为辅助判断）
        self.importance_patterns = [
            r'\b(我|你)\s*(是|叫|的名字是)\s*\S+',  # 名字
            r'\d{1,3}\s*岁',  # 年龄
            r'\b(生日|出生|生日的)\b',  # 生日
            r'\b(住|住在|来自|家乡)\s*\S+',  # 居住地
            r'\b(喜欢|爱|热爱|讨厌|害怕|担心)\s*\S+',  # 喜好/恐惧
            r'\b(重要|关键|务必|一定|必须|永远|承诺)\b',  # 重要事件
            r'\b(朋友|家人|父母|姐妹|兄弟|伴侣|恋人|同事)\b',  # 关系
            r'\b(职业|工作|公司|学校|专业|学习|研究)\b',  # 职业
            r'\b(每天|每周|每月|经常|总是|从未|从不)\s*\S+',  # 习惯
        ]

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的 embedding"""
        if self.use_embedding and self.embedding_model:
            return self.embedding_model.encode(text, normalize_embeddings=True)
        else:
            # 如果不使用 embedding，返回空数组
            return np.array([])

    def _calculate_importance_score(self, text: str) -> float:
        """
        基于规则计算信息的重要性分数 (0-1)
        """
        score = 0.0

        # 1. 基于关键词模式的分数
        for pattern in self.importance_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15
                if score >= 0.8:
                    break

        # 2. 基于句子长度和复杂度的分数
        word_count = len(text)
        if 5 <= word_count <= 50:
            score += 0.1
        elif word_count > 50:
            score += 0.15

        # 3. 包含数字
        if re.search(r'\d+', text):
            score += 0.1

        # 4. 包含问句（可能是重要询问）
        if re.search(r'[？?]', text):
            score += 0.05

        # 5. 包含感叹号（强调）
        if re.search(r'[！!]', text):
            score += 0.05

        return min(score, 1.0)

    def _is_factual_statement(self, text: str) -> bool:
        """
        判断是否是一个陈述性事实
        """
        factual_patterns = [
            r'^我\s*(是|叫|有|在|住|来自|喜欢|爱|讨厌|害怕|做|工作|学习)',
            r'^你\s*(是|叫|有|在|住|来自)',
            r'^我的\s*(名字|年龄|生日|职业|爱好|朋友|家人|工作)',
            r'\b(事实上|实际上|确实|真的)\b',
        ]

        for pattern in factual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def extract_memories(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """
        从对话中提取重要的记忆信息
        """
        combined_text = f"用户说: {user_message}\nAI说: {ai_response}"

        # 分割成独立的语句
        sentences = re.split(r'[。！？.!?，,;；\n]+', combined_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        extracted_memories = []

        for sentence in sentences:
            # 计算重要性分数
            importance_score = self._calculate_importance_score(sentence)

            # 如果是陈述性事实且重要性超过阈值
            if (self._is_factual_statement(sentence) or
                    importance_score > self.importance_threshold):

                # 判断是否已有相似记忆（使用文本匹配或 embedding）
                is_duplicate = False

                if self.use_embedding:
                    # 获取 embedding
                    embedding = self._get_embedding(sentence)

                    for existing in self.memories:
                        if 'embedding' in existing and len(existing['embedding']) > 0:
                            existing_embedding = existing['embedding']
                            similarity = cosine_similarity(
                                [embedding],
                                [existing_embedding]
                            )[0][0]

                            if similarity > self.similarity_threshold:
                                is_duplicate = True
                                # 更新已有记忆的重要性分数
                                if importance_score > existing['importance_score']:
                                    existing['importance_score'] = importance_score
                                    existing['timestamp'] = datetime.now().isoformat()
                                break
                else:
                    # 不使用 embedding 时，用文本匹配去重
                    for existing in self.memories:
                        if existing['content'] == sentence:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    # 确定信息类型
                    info_type = self._classify_information(sentence)

                    memory_item = {
                        'content': sentence,
                        'type': info_type,
                        'importance_score': importance_score,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'user' if '用户说' in sentence[:10] else 'ai',
                    }

                    # 如果使用 embedding，添加 embedding
                    if self.use_embedding:
                        memory_item['embedding'] = self._get_embedding(sentence)

                    extracted_memories.append(memory_item)

        return extracted_memories

    def _classify_information(self, text: str) -> str:
        """对信息进行分类"""
        # 个人身份
        if re.search(r'\b(名字|叫|姓名|称呼)\b', text):
            return '个人身份'
        elif re.search(r'\b(年龄|岁|生日|出生|年月|日期)\b', text):
            return '年龄/生日'
        elif re.search(r'\b(住|居住|家乡|城市|地址|地方)\b', text):
            return '居住地'

        # 情感和偏好
        if re.search(r'\b(喜欢|爱|热爱|欣赏|钟爱)\b', text):
            return '喜好'
        elif re.search(r'\b(讨厌|不喜欢|害怕|恐惧|担心|忧虑)\b', text):
            return '担忧/恐惧'

        # 职业和教育
        if re.search(r'\b(工作|职业|公司|职位|老板|同事)\b', text):
            return '职业信息'
        elif re.search(r'\b(学校|学习|研究|专业|课程|考试)\b', text):
            return '教育信息'

        # 人际关系
        if re.search(r'\b(朋友|家人|父母|兄弟|姐妹|恋人|伴侣|闺蜜)\b', text):
            return '人际关系'

        # 重要事件
        if re.search(r'\b(重要|关键|务必|必须|承诺|约定)\b', text):
            return '重要事件'

        # 习惯和日常
        if re.search(r'\b(每天|每周|每月|经常|总是|从不|习惯)\b', text):
            return '日常习惯'

        return '一般信息'

    def add_memories(self, new_memories: List[Dict[str, Any]]):
        """添加新的记忆，自动去重和更新"""
        for memory in new_memories:
            if memory['importance_score'] < self.importance_threshold:
                continue

            # 检查是否已存在相似记忆
            is_duplicate = False

            if self.use_embedding and 'embedding' in memory:
                for existing in self.memories:
                    if 'embedding' in existing and len(existing['embedding']) > 0:
                        similarity = cosine_similarity(
                            [memory['embedding']],
                            [existing['embedding']]
                        )[0][0]
                        if similarity > self.similarity_threshold:
                            is_duplicate = True
                            # 更新信息
                            if memory['importance_score'] > existing['importance_score']:
                                existing['importance_score'] = memory['importance_score']
                                existing['content'] = memory['content']
                                existing['timestamp'] = datetime.now().isoformat()
                            break
            else:
                # 文本匹配去重
                for existing in self.memories:
                    if existing['content'] == memory['content']:
                        is_duplicate = True
                        break

            if not is_duplicate:
                self.memories.append(memory)

    def get_most_important_memories(self, top_k: int = 15, min_score: float = 0.5) -> str:
        """
        获取最重要的记忆，格式化为文本
        """
        # 按重要性排序
        sorted_memories = sorted(
            self.memories,
            key=lambda x: x['importance_score'],
            reverse=True
        )

        # 过滤掉低分记忆
        filtered = [m for m in sorted_memories if m['importance_score'] >= min_score]

        if not filtered:
            return ""

        # 按类型分组
        grouped = defaultdict(list)
        for memory in filtered[:top_k]:
            grouped[memory['type']].append(memory)

        memory_text = "\n【我记住的重要信息】\n"
        for info_type, memories_list in grouped.items():
            memory_text += f"\n{info_type}:\n"
            for memory in memories_list[:5]:
                memory_text += f"  • {memory['content']}\n"
                if memory['importance_score'] > 0.8:
                    memory_text += f"    ⭐ 非常重视\n"

        return memory_text

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        # 移除 embedding 字段以便显示
        clean_memories = []
        for m in self.memories:
            clean_m = {k: v for k, v in m.items() if k != 'embedding'}
            clean_memories.append(clean_m)
        return sorted(clean_memories, key=lambda x: x['importance_score'], reverse=True)

    def clear_memories(self):
        """清空所有记忆"""
        self.memories = []


def keep_recent_memories(memories, max_pairs: int = 10):
    """保留最近n轮对话"""
    system_messages = [m for m in memories if isinstance(m, SystemMessage)]
    conversation_messages = [m for m in memories if not isinstance(m, SystemMessage)]

    if len(conversation_messages) > max_pairs * 2:
        recent_messages = conversation_messages[-max_pairs * 2:]
    else:
        recent_messages = conversation_messages

    return system_messages + recent_messages


# 读取配置文件
load_dotenv(dotenv_path="C:/Users/21136/anaconda3/envs/ml starting/envir_1.env", override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 初始化模型
model = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    temperature=2,
    timeout=240,
)

# 初始化智能记忆管理器（使用免费本地模型）
memory_manager = IntelligentMemoryManager(
    model_name="all-MiniLM-L6-v2",  # 使用小模型，下载更快
    similarity_threshold=0.7,
    importance_threshold=0.5
)

# 系统提示
system_prompt = """你现在是一只可爱的猫娘，你是我的女友。
你叫萌萌。

重要提示：你会自动记住用户告诉你的重要信息，并在对话中自然地提及这些信息。
记住要真诚、温暖，像真正的女友一样关心主人。"""

# 初始化对话
memories = [SystemMessage(content=system_prompt)]

print("=" * 60)
print("💕 智能猫娘已启动 - 正在加载记忆系统...")
print("=" * 60)

conversation_round = 0

while True:
    user_input = input("\n主人: (按 q 退出) ")
    if user_input.lower() == 'q':
        break

    conversation_round += 1

    # 添加用户消息
    user_message = HumanMessage(content=user_input)
    memories.append(user_message)

    try:
        # 获取AI回复
        response = model.invoke(memories)
        ai_response = response.content

        print(f"萌萌: {ai_response}")

        # 添加AI回复到对话历史
        ai_message = AIMessage(content=ai_response)
        memories.append(ai_message)

        # 每2轮对话进行一次记忆提取（减少计算开销）
        if conversation_round % 2 == 0:
            # 智能提取重要信息
            new_memories = memory_manager.extract_memories(user_input, ai_response)
            if new_memories:
                memory_manager.add_memories(new_memories)

                # 更新系统提示中的记忆信息
                updated_memory_text = memory_manager.get_most_important_memories(top_k=10)
                if updated_memory_text:
                    # 重建系统消息
                    system_msgs = [m for m in memories if isinstance(m, SystemMessage)]

                    # 更新或添加记忆系统消息
                    if len(system_msgs) >= 2:
                        system_msgs[1] = SystemMessage(content=updated_memory_text)
                    else:
                        system_msgs.append(SystemMessage(content=updated_memory_text))

                    # 保留最近的对话
                    conversation_msgs = [m for m in memories if not isinstance(m, SystemMessage)]
                    if len(conversation_msgs) > 8:
                        conversation_msgs = conversation_msgs[-8:]

                    memories = system_msgs + conversation_msgs

        # 限制短期记忆（保留最近2轮）
        memories = keep_recent_memories(memories, 2)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 60)
print("对话结束。")
print("\n长期记忆摘要:")

all_memories = memory_manager.get_all_memories()
if all_memories:
    print(f"\n共记录了 {len(all_memories)} 条重要信息：\n")
    for i, memory in enumerate(all_memories[:20], 1):
        importance_level = "⭐" * int(memory['importance_score'] * 5)
        print(f"{i}. [{memory['type']}] {memory['content']}")
        print(f"   重要性: {memory['importance_score']:.2f} {importance_level}")
        print()
else:
    print("暂无重要记忆。")

# 保存记忆到文件
try:
    with open('memories_backup.json', 'w', encoding='utf-8') as f:
        json.dump(all_memories, f, ensure_ascii=False, indent=2)
    print(f"\n记忆已保存到 memories_backup.json (共 {len(all_memories)} 条)")
except Exception as e:
    print(f"保存记忆时出错: {e}")
