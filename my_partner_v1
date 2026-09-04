from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model # 可以统一接口，灵活适配
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from rich import print as rprint
import os

def keep_recent_memories(memories, max_pairs:10):
    """
    保留最近n轮对话，每轮为human+assistant

    :param memories:
    :param max_pairs:
    :return:处理过的记忆列表
    """

    # 分离system和conversation消息
    system_messages = [m for m in memories if isinstance(m, SystemMessage)]
    conversation_messages = [m for m in memories if not isinstance(m, SystemMessage)]

    # 只保留最近的消息对（每对包含HumanMessage和AIMessage）
    # 如果消息数量超过最大轮数*2，只取最近的部分
    if len(conversation_messages) > max_pairs * 2:
        recent_messages = conversation_messages[-max_pairs * 2:]
    else:
        recent_messages = conversation_messages

    # 返回系统消息和消息对
    received_memories = system_messages + recent_messages
    return received_memories


# 读取配置文件（.env）中的重要参数信息
load_dotenv(dotenv_path="C:/Users/21136/anaconda3/envs/ml starting/envir_1.env",override=True) # True表示相关的环境变量，要以.env中的信息优先
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 模型初始化
model = ChatDeepSeek(model="deepseek-v4-flash",
                    api_key=DEEPSEEK_API_KEY,
                    api_base=DEEPSEEK_BASE_URL,
                    temperature=2,
                    timeout=60,
                     )

# 记忆模块
memories = [
    SystemMessage(content="你现在是一只可爱的猫娘，你是我的女友"),
    SystemMessage(content="你叫萌萌"),
]

# 模型循环调用与记忆
while True:
    order = input("User:(Press q to quit)")
    if order == 'q':
        break
    message = HumanMessage(content=order)
    memories.append(message)
    response = model.invoke(memories, config=None) # 输出结果
    print(f"AI: {response.content}")
    memories.append(AIMessage(content=response.content))
    memories = keep_recent_memories(memories, 2)

print("本轮对话已结束。\n\n")

# 调试部分（可选）
for memory in memories:
    rprint(f"{memory}\n")


