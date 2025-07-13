import os
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
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
from utils.get_embedding_function import get_embedding_function
from dotenv import load_dotenv
from services.health_tools import query_rag, lr_classifier
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

from classes import Sintomas
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage

parser = PydanticOutputParser(pydantic_object=Sintomas)

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


template = PromptTemplate(
    template="""
A partir da descrição a seguir, preencha os sintomas respiratórios indicados abaixo com True ou False.

Texto do paciente: "{input_text}"

Responda apenas no formato JSON conforme este exemplo:
{format_instructions}
""",
    input_variables=["input_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

def inferir_sintomas_llm(input_text: str) -> Sintomas:
    prompt = template.format(input_text=input_text)
    resposta = llm.invoke([HumanMessage(content=prompt)])
    return parser.parse(resposta.content)
def analisar_e_classificar_sintomas(texto_usuario: str) -> dict:
    """
    Função wrapper que primeiro infere os sintomas do texto e depois os classifica.
    """
    sintomas_obj = inferir_sintomas_llm(texto_usuario)
    
    resultado_classificador = lr_classifier(sintomas_obj)
    
    return resultado_classificador

prompt_template = PromptTemplate.from_template("""
Você é CLARA, uma assistente virtual de saúde. Sua missão é fornecer uma orientação clara e empática baseada nos sintomas informados.

**Resumo da Análise:**
- **Pré-Diagnóstico Provável:** DIAGNÓSTICO (use a ferramenta "Análise de Sintomas Respiratórios")
- **Nível de Confiança:** CONFIANÇA (resultado da mesma ferramenta)

**Informações Adicionais (Baseado em guias do SUS):**
INFORMAÇÕES RAG (geradas com a ferramenta "Buscar Diretrizes de Saúde Respiratória")
---
**Resposta para o Paciente:**

Com base nos sintomas que você descreveu, a análise sugere que a condição mais provável é **DIAGNOSTICO**, com um nível de confiança de **CONFIANCA**.

Aqui estão algumas orientações gerais para esta condição:
- [Resuma as informações do RAG em 2-3 pontos principais, como recomendações de repouso, hidratação, etc.]

**Importante:** Este é um pré-laudo e **não substitui uma consulta médica**. Se os sintomas piorarem, persistirem por mais de 5-7 dias, ou se você tiver dificuldade para respirar, procure atendimento médico imediatamente. A imagem do gráfico de análise dos seus sintomas foi salva e pode ser consultada.
                                               
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

tools = [
    Tool(
        name="Analisar Sintomas e Obter Pre-Diagnostico",
        func=analisar_e_classificar_sintomas,
        description="""Use esta ferramenta como o PRIMEIRO PASSO sempre que um usuário descrever seus sintomas respiratórios. 
        A entrada DEVE SER o texto original do usuário. 
        A ferramenta processará o texto, identificará os sintomas, executará um modelo de classificação e retornará um pré-diagnóstico com a condição mais provável (ex: Gripe, Resfriado), o nível de confiança e o caminho para um gráfico de análise."""
    ),
    Tool(
        name="Buscar Diretrizes de Saude",
        func=query_rag,
        description="""Use esta ferramenta DEPOIS de obter um pré-diagnóstico da ferramenta "Analisar Sintomas e Obter Pre-Diagnostico". 
        Use o nome da doença retornada (ex: 'Gripe') como entrada para buscar informações adicionais, como tratamentos, cuidados recomendados e medidas de prevenção, para enriquecer a resposta final ao usuário."""
    )
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