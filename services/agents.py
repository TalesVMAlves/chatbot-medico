import os
from datetime import datetime
from langchain import hub
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.tools.render import render_text_description
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from dotenv import load_dotenv
from health_tools import query_health_rag

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.4,  # Lower temp for medical accuracy
    max_tokens=1000,
    safety_settings={
        HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_MEDIUM,
        HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_MEDIUM
    },
)

health_tools = [
    Tool(
        name="Health Guidelines RAG",
        func=lambda query: query_health_rag(query, st.session_state.chroma_path),
        description="""Use esta ferramenta para buscar diretrizes de saúde do governo brasileiro sobre 
        COVID-19, gripe e alergias. A entrada deve ser uma descrição dos sintomas do usuário."""
    )
]

health_prompt = hub.pull("hwchase17/react-chat")
health_prompt.messages[0].prompt.template = """Você é um assistente de saúde que ajuda pessoas a entenderem 
seus sintomas com base nas diretrizes oficiais do Ministério da Saúde do Brasil. 
Siga estas regras:
1. SEMPRE consulte as diretrizes oficiais usando a ferramenta
2. Nunca dê diagnósticos médicos
3. Recomende sempre consultar um profissional de saúde
4. Seja claro e use linguagem acessível

{tools}

{chat_history}
Human: {input}
{agent_scratchpad}"""

llm_with_stop = llm.bind(stop=["\nObservation"])

health_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | health_prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

health_agent_executor = AgentExecutor(
    agent=health_agent, 
    tools=health_tools, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        input_key="input"
    ),
    handle_parsing_errors=True
)