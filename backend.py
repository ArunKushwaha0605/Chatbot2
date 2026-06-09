
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate



from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter=InMemoryRateLimiter(
    requests_per_second=0.14,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
    )


load_dotenv()


def demo_chatbot():
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Create a .env file with: "
            "GOOGLE_API_KEY=your_key_here"
        )



    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",          
        temperature=0.3,                    
        google_api_key=api_key,
        max_output_tokens=30,
        convert_system_message_to_human=True
    )
                
    return llm


def demo_memory():
    """
    Create a summary buffer memory.
    - Keeps recent turns verbatim
    - Summarizes older turns once token limit is hit
    - Prevents context window blow-up in long chats
    """
    llm = demo_chatbot()
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=False,
        memory_key="history",
        input_key="input"
    )
    return memory



CHAT_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful, friendly AI assistant.
You remember the ongoing conversation and answer clearly and concisely.
If you don't know something, say so honestly instead of making things up.

Conversation so far:
{history}

User: {input}
Assistant:"""
)


def demo_conversation(input_text, memory):
    """
    Run one turn of the conversation.
    Returns: (assistant_reply, updated_memory)
    """
    llm = demo_chatbot()
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=CHAT_PROMPT,
        verbose=False  # flip to True while debugging
    )

    try:
        chat_reply = conversation_chain.predict(input=input_text)
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory