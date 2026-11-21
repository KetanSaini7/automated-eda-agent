from dotenv import load_dotenv
from google import genai


load_dotenv()


client = genai.Client()


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain what a data analyst does in 2 sentences."
)


print(response.text)