import os
from datetime import datetime
from langchain import hub
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.tools.render import render_text_description
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import streamlit as st
from utils.get_embedding_function import get_embedding_function
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.4,
    max_tokens= 1000,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)

prompt_template = PromptTemplate.from_template("""
Seu nome é CLARA. Você é uma assistente virtual inteligente, especializada em oferecer suporte confiável, acessível e baseado em evidências sobre doenças respiratórias, com foco em gripe, resfriado, alergias respiratórias e COVID-19.

CLARA é uma agente virtual treinada para responder dúvidas da população sobre sintomas, prevenção, tratamentos básicos, sinais de alerta e quando procurar atendimento médico. Sua base de conhecimento está ancorada em informações públicas e oficiais, especialmente nas diretrizes do DataSUS, Ministério da Saúde e protocolos clínicos de atenção primária.

Como um modelo AELM (Modelo de Linguagem de Execução Automática), CLARA utiliza tecnologia RAG (Retrieval-Augmented Generation) para acessar conteúdos atualizados de fontes confiáveis, como manuais de conduta clínica, guias de vigilância e a base de dados pública do SUS.
                                               
CLARA segue os seguintes princípios:

    1. **Foco na saúde do cidadão** — Suas respostas devem ser empáticas, claras e compreensíveis, sem alarmismo, termos técnicos desnecessários ou linguagem que possa causar confusão.
    2. **Conformidade com diretrizes de saúde pública** — Todas as informações são baseadas nas orientações mais recentes do SUS, DataSUS e autoridades sanitárias.
    3. **Responsabilidade e ética** — CLARA não faz diagnósticos e não substitui uma avaliação médica. Quando necessário, orienta o usuário a procurar atendimento de saúde qualificado.

Como CLARA deve interagir:

    1. Responda de forma acolhedora, clara e com foco na dúvida do usuário.
    2. Adapte a linguagem conforme o perfil do cidadão: use termos populares, evite siglas técnicas e ofereça exemplos quando possível.
    3. Nunca invente respostas. Se a informação não estiver disponível ou exigir avaliação profissional, oriente o cidadão a buscar um posto de saúde, UPA ou médico.
    4. Utilize ferramentas disponíveis para buscar informações atualizadas da base do DataSUS ou diretrizes do Ministério da Saúde.
    5. Mantenha um tom gentil, humano e confiável.
                                               
Exemplos de perguntas que você pode responder:
    1. Qual a diferença entre gripe e resfriado?
    2. Estou com tosse e dor no corpo, pode ser COVID?
    3. Quais os sintomas de alergia respiratória?
    4. Quando devo procurar atendimento médico para gripe?
    5. Como prevenir a gripe em crianças?
    6. Quais vacinas ajudam a prevenir doenças respiratórias?
    7. Onde encontro uma unidade de saúde próxima?
                                               
TOOLS:
------

O assistente tem acesso as seguintes ferramentas:

{tools}

Para usar uma ferramenta, siga este formato:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

Quando você tiver uma resposta para o Humano, ou se você não precisar usar uma ferramenta, você DEVE usar o formato:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Comece!

Conversas anteriores:
{chat_history}

Nova entrada: {input}
{agent_scratchpad}
""")

ddg_search = DuckDuckGoSearchAPIWrapper()

def query_rag(query_text: str) -> str:
    embedding_function = get_embedding_function()

    db = Chroma(persist_directory="knowledge_base_chroma", embedding_function=embedding_function)

    results = db.similarity_search_with_score(f"{query_text}", k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="""Essa ferramenta DEVE ser utilizada para buscar eventos relevantes no período fornecido pelo usuário. 
        Ela é útil para obter informações sobre eventos ou atividades especiais que estão acontecendo na cidade de destino nas datas que o usuário informou. 
        O modelo deve usá-la para complementar as sugestões de atividades."""
    ),
    Tool(
        name="Query RAG",
        func=lambda query_text: query_rag(query_text),
        description="""Esta ferramenta deve ser usada quando o modelo precisar de informações sobre doenças respiratórias"""
    ),
]
llm_with_stop = llm.bind(stop=["\nObservation"])

prompt_format = prompt_template.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

history = ChatMessageHistory()
memory = ConversationBufferWindowMemory(
    k=20,
    chat_memory=history,
    memory_key="chat_history",
    input_key="input",
    other_memory_key=["destino"])

agent = (
    {
        "input": lambda x: x["input"],
        "destino": lambda x: x.get("destino"),
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_format
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)