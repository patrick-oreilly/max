from langchain_classic.memory import ConversationBufferMemory

def get_memory():
    return ConversationBufferMemory(return_messages=True, memory_key="chat_history")