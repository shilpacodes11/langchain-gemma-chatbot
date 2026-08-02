from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it"
)

print("Gemma Chatbot")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    response = model.invoke(user)

    print("Gemma:", response.text)
    print()