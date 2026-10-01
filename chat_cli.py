from chatbot.rag import Chatbot

bot = Chatbot()
print("Student support bot. Type 'quit' to exit.")
while (q := input("\nYou: ").strip()) and q.lower() != "quit":
    r = bot.ask(q)
    print(f"\nBot [{r['topic']}]: {r['answer']}\nSources: {', '.join(r['sources'])}")
