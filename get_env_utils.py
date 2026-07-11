from dotenv import load_dotenv
import os
load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_url = os.getenv("DEEPSEEK_API_URL")
city_key = os.getenv("OPENWEATHER_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")
travily_api_key = os.getenv("TRAVILY_API_KEY")