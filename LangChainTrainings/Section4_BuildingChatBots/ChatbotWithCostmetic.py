from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import streamlit as st

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///chathistory.db")

user_id = "Karthik"

with st.sidebar:
    st.image("./EA-Logo.png", width=150)
    user_id = st.text_input("Enter your name", user_id)
    role = st.radio("How detailed should your answer be?", ["Beginner", "Expert", "PhD"], index=0)
    if st.button("Start new chat"):
        st.session_state.chat_history = []
        get_session_history(user_id).clear()
        
st.markdown(
    """
    <div style='display: flex; height: 70vh; justify-content: center; align-items: center;'>
        <h2>What can I help you today?</h2>
    </div>
    """,
    unsafe_allow_html=True
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
       st.markdown(message['content'])


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5:latest",
    temperature=0.5,
    max_tokens=250,
)

template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ('system', f"You are an {role} level user to answer this query"),
    ('human', "{prompt}")
])

chain = template | llm | StrOutputParser()

def invoke_history(chain, session_id, prompt):
    history = RunnableWithMessageHistory(chain, 
                                         get_session_history,
                                         input_messages_key="prompt",
                                         history_messages_key="history")
    
    for response in history.stream({"prompt": prompt},config={"configurable": {"session_id": session_id}}):
        yield response

prompt = st.chat_input("Enter your query")

if prompt:
    st.session_state.chat_history.append({'role': 'user', "content": prompt})

    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message('assistant'):
        streamResponse = st.write_stream(invoke_history(chain, user_id, prompt))

    st.session_state.chat_history.append({'role': 'assistant', "content": streamResponse})
