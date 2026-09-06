from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv

import streamlit as st
import os

load = load_dotenv(".env")

model_name = "qwen3:8b"

llm = ChatOllama(
    base_url="http://localhost:11434",
    model=model_name,
    temperature=0.5,
    num_predict=250
)
session_id = "Gk"

def get_session_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chathistory.db",
    )

st.title('Hey how can i help you! ')
st.write('Enter your query')

session_id = st.text_input("Enter your name ", session_id)

if st.button("start new session"):
    st.session_state.chat_history = []
    get_session_history(session_id).clear()
    st.success(f"New session started with session id: {session_id}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history =[]
    
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# Prompt with history
template = ChatPromptTemplate.from_messages([
    ("human", "{prompt}"),
    ("placeholder", "{history}"),
])

chain = template | llm | StrOutputParser()

history = RunnableWithMessageHistory(chain,get_session_history,input_messages_key="prompt",history_messages_key="history",)

# get_session_history(session_id).clear()


prompt = st.chat_input("Enter your question")

st.session_state.chat_history = []

if prompt:
    st.session_state.chat_history.append({'role': 'user',"content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner(f'Thinking... The local {model_name} is preparing a response.'):
            response = history.invoke(
                {"prompt": prompt},
                config={"configurable": {"session_id": session_id}},
            )
        st.session_state.chat_history.append({'role': 'assistant', "content": response})
        st.markdown(response)



