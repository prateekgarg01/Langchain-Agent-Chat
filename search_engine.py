import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()


# Arxiv

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Wikipedia
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun( api_wrapper=api_wrapper_wiki)

search=DuckDuckGoSearchRun(name="Search")

st.title("Langchain - Chat with Search")
"""
Chat is implemented with **Agents** like - **Wikipedia**, **Arxiv** and **DuckDuckGo** along with 
**'StreamlitCallbackHandler'** to display the thoughts and actions of an agent in an interactive Streamlit app. 
Chat is also utilizing **GROQ LLAMA3** LLM model.
"""

# Sidebar for Settings
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",type="password")
#os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. how can I help You."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=groq_api_key,model="llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=False)


    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.success(response)
