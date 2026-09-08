from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rich import print as rprint
import os

# 读取配置文件
load_dotenv(dotenv_path="C:/Users/21136/anaconda3/envs/ml starting/envir_1.env", override=True)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 模型初始化
model = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    temperature=2,
    timeout=60
)


class WeatherSchema(BaseModel):
    city: str = Field(default="Beijing", description="specific city name")
    if_forcast: bool = Field(default=False, description="if weather forecast is available")


class StockSchema(BaseModel):
    company: str = Field(default="Google", description="specific company name")
    stock_price: float = Field(default=1.0, description="stock's price")


@tool(description="获取具体城市天气情况，可以包含天气预报", args_schema=WeatherSchema)
def get_weather(city: str, if_forcast: bool) -> str:
    """获取天气信息"""
    ret = f"{city}: it's sunny."
    if if_forcast:
        ret = ret + f"\n{city}: it'll be sunny tomorrow."
    return ret


@tool(description="Get specific company's stock price.", args_schema=StockSchema)
def get_stock_price(company: str) -> str:
    """获取股票价格"""
    # 修正：使用字典而不是列表
    company_stocks = {
        "Apple": 16.0,
        "Google": 12.0,
        "Microsoft": 14.0,
        "Huawei": 11.0,
    }

    if company in company_stocks:
        # 修正：返回实际股价，而不是传入的参数
        actual_price = company_stocks[company]
        return f"Information: {company}: {actual_price}USD"
    else:
        return f"Not Found {company}."


# 工具与模型结合
model_with_tools = model.bind_tools([get_weather, get_stock_price])

# 消息列表
# 消息列表
messages = [
    HumanMessage("今天杭州天气如何？明天呢？"),
    HumanMessage("今天华为公司股价多少？和谷歌相比呢？"),
    HumanMessage("评估一下不同企业的股价情况。"),
    HumanMessage("亚马逊公司股价多少？"),
]

while True:
    response = model_with_tools.invoke(messages)

    messages.append(response)
    if not response.tool_calls:
        print("未找到答案\n")
        break

    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_stock_price":
            stock_result = get_stock_price.invoke(tool_call)
            print("stock_result:", stock_result)
            messages.append(stock_result)
        if tool_call["name"] == "get_weather":
            weather_result = get_weather.invoke(tool_call)
            print("weather_result:", weather_result)
            messages.append(weather_result)

for message in messages:
    rprint(message)
