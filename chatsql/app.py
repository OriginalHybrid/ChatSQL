import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os
import re
import pandas as pd

st.set_page_config(page_title="LangChain: ChatSQL", page_icon="🦜")
st.title("LangChain: ChatSQL")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"


def process_sql_response(response):
    print("Processing response...")
    print(response)
    # Extract SQL query
    sql_match = re.search(r'SELECT.*?;', response, re.IGNORECASE | re.DOTALL)
    
    # Extract table data
    table_match = re.search(r'((?:^|\n)(?:\w+\s*\|\s*)+\w+\s*\n(?:[-|]+\s*\n)?(?:(?:\w+\s*\|\s*)+\w+\s*\n?)+)', response, re.MULTILINE)
    
    if sql_match and table_match:
        sql_query = sql_match.group(0).strip()
        table_data = table_match.group(1).strip()
        
        markdown_output = f"""
        ## SQL Query:
        {sql_query}
        ## Output table:
        {table_data}
        """
        st.markdown(markdown_output)
    else:
        # If no SQL query or table found, display the entire response
        st.markdown(response)

radio_opt=["Use SQLLite 3 Database (.db file)","Connect to you MySQL Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Host")
    mysql_user=st.sidebar.text_input("MYSQL User")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MySQL database")
else:
    # Display file uploader
    uploaded_file = st.sidebar.file_uploader("Choose your SQLite database file", type=["db"])
    
    if uploaded_file is not None:
        # Get the file path
        df_file_path = os.path.join(os.getcwd(), uploaded_file.name)
        
        # Save the uploaded file
        with open(df_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.sidebar.success(f"File uploaded successfully. Path: {df_file_path}")
    
    db_uri=LOCALDB

HF_TOKEN=st.sidebar.text_input(label="Hugging Face Token",type="password")

if not db_uri:
    st.info("Please enter the database information and uri")

if not HF_TOKEN:
    st.info("Please add the Hugging Face Token")

else:

    ## LLM model
    # llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    import os
    print("HF Token", HF_TOKEN)
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN


    from langchain import PromptTemplate
    from langchain_huggingface import HuggingFaceEndpoint
    model = "meta-llama/Llama-3.2-1B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=model,
        max_length=256,
        temperature=0.5,
    )

    @st.cache_resource(ttl="2h")
    def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
        if db_uri==LOCALDB:
            if df_file_path:
                st.write(f"Selected database file: {df_file_path}")
            # dbfilepath=(Path(__file__).parent/"student.db").absolute()
            print("dbfilepath", df_file_path)
            # sqlite_uri = 'sqlite:///' +dbfilepath
            # db = SQLDatabase.from_uri(sqlite_uri)
            creator = lambda: sqlite3.connect(f"file:{df_file_path}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri==MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   
        
    if db_uri==MYSQL:
        db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
    else:
        db=configure_db(db_uri)

    from langchain_core.prompts import ChatPromptTemplate

    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)


    def get_schema(_):
        print(type(db))
        schema = db.get_table_info()
        return schema


    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query=st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            # streamlit_callback=StreamlitCallbackHandler(st.container())
            response=sql_chain.invoke({"question": user_query})
            process_sql_response(response)
            st.session_state.messages.append({"role":"assistant","content":response})
            # st.write(response)

            


