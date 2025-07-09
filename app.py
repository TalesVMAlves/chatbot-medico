import streamlit as st
from services.agents import agent_executor, llm, memory
from langchain.schema import AIMessage, HumanMessage

st.set_page_config(
    page_title="Assistente Virtual -- Doenças Respiratórias",
    page_icon="🤖",
    layout="centered"
)

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.selected_destino = "UFRN"
    st.session_state.chroma_city_path = "./chroma_db"

st.title("🤖 Assistente Virtual -- Doenças Respiratórias")
st.caption("Sua assistente virtual para ajuda a prevenção e cuidados de doenças respiratórias")

for msg in st.session_state.messages:
    avatar = "👤" if isinstance(msg, HumanMessage) else "🤖"
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant", avatar=avatar):
        st.markdown(msg.content)

if prompt := st.chat_input("Como posso ajudar?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                output = response.get("output", "Desculpe, ocorreu um erro.")
            except Exception as e:
                output = f"Erro no sistema: {str(e)}"
            
            st.markdown(output)
    
    st.session_state.messages.append(AIMessage(content=output))
    
    memory.save_context({"input": prompt}, {"output": output})