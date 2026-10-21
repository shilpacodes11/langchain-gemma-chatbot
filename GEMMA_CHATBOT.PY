
import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MEMORY_FILE = "chat_memory.json"

model = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it"
)

# Load previous conversations from the file
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        history = json.load(file)
else:
    history = []

print("Gemma Chatbot with Memory")
print("Type 'exit' to quit.")
print("Type 'clear' to erase saved memory.\n")

while True:
    user = input("You: ").strip()

    if user.lower() == "exit":
        break

    if not user:
        continue

    if user.lower() == "clear":
        history = []
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
        print("Memory cleared!\n")
        continue

    # Include previous messages in the prompt
    previous_chats = history[-20:]

    conversation = ""
    for message in previous_chats:
        if message["role"] == "user":
            conversation += f'User: {message["content"]}\n'
        else:
            conversation += f'Gemma: {message["content"]}\n'

    prompt = f"""
You are a helpful chatbot with conversation memory.

Use the previous conversation to understand the user's
context and answer follow-up questions naturally.
Do not claim to remember something that is not in the
provided conversation history.

Previous conversation:
{conversation}

Current user message:
{user}

Reply naturally to the user.
"""

    response = model.invoke(prompt)
    answer = response.text

    print("Gemma:", answer, "\n")

    # Save both sides of the conversation
    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": answer})

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
