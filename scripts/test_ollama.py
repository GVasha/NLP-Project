from ollama import chat

response = chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(response["message"]["content"])