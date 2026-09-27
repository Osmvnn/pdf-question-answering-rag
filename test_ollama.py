from ollama import Client

client = Client(host="http://localhost:11434", timeout=180.0)

context = "The library closes at 6 PM on Fridays."

questions = [
    "When does the library close on Friday?",
    "Who manages the library?",
]

for question in questions:
    response = client.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only using the supplied context. "
                    "If the information is missing, say: "
                    "'The document does not provide this information.' "
                    "Keep your answer short."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        options={"temperature": 0},
    )

    print(f"\nQuestion: {question}")
    print(f"Answer: {response.message.content}")