from langchain.chat_models import init_chat_model
from get_env_utils import deepseek_api_key, deepseek_api_url, city_key, weather_api_key, travily_api_key

deepseek_chat_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=deepseek_api_key,    
    base_url=deepseek_api_url,
)

deepseek_flash_model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=deepseek_api_key,    
    base_url=deepseek_api_url,
)

# deepseek_pro_model = init_chat_model(
#     model="deepseek-v4-pro",
#     model_provider="deepseek",
#     api_key=deepseek_api_key,    
#     base_url=deepseek_api_url,
# )
if __name__ == "__main__":
    response = deepseek_chat_model.invoke("你好")
    print(response.content)
