import streamlit as st
import json
from services.agents import agent_executor, inferir_sintomas_llm
from langchain.schema import AIMessage, HumanMessage

st.set_page_config(
    page_title="CLARA - Assistente de Saúde Respiratória",
    page_icon="🤖",
    layout="centered"
)

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        AIMessage(content="Olá! Sou a CLARA, sua assistente de saúde. Por favor, descreva seus sintomas para que eu possa ajudar.")
    )

st.title("🤖 CLARA - Assistente de Saúde Respiratória")
st.caption("Uma IA para pré-análise de sintomas, baseada em diretrizes do SUS.")

for msg in st.session_state.messages:
    avatar = "👤" if isinstance(msg, HumanMessage) else "🤖"
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant", avatar=avatar):
        st.markdown(msg.content)
        if isinstance(msg, AIMessage) and "image_path" in msg.additional_kwargs:
            st.image(msg.additional_kwargs["image_path"], caption="Análise de Influência dos Sintomas (SHAP)")

if prompt := st.chat_input("Descreva seus sintomas aqui..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analisando seus sintomas e consultando as diretrizes..."):
            try:
                sintomas_obj = inferir_sintomas_llm(prompt)
                
                response = agent_executor.invoke({"input": f"Analise os seguintes sintomas: {sintomas_obj.dict()}"})
                
                output = response.get("output", "Desculpe, ocorreu um erro.")
                
                output_text = output
                image_path = "artefatos/shap_force.png"

                try:
                    if isinstance(output, str) and "texto_resultado" in output:
                        output_dict = eval(output) 
                    elif isinstance(output, dict):
                        output_dict = output
                    else:
                        output_dict = {}

                    if "texto_resultado" in output_dict:
                        output_text = output_dict["texto_resultado"]
                        image_path = output_dict.get("caminho_imagem")
                except (SyntaxError, NameError, TypeError):
                    output_text = str(output)

                st.markdown(output_text)
                if image_path:
                    st.image(image_path, caption="Análise de Influência dos Sintomas (SHAP)")
                
                st.session_state.messages.append(
                    AIMessage(
                        content=output_text,
                        additional_kwargs={"image_path": image_path} if image_path else {}
                    )
                )

            except Exception as e:
                error_message = f"Desculpe, ocorreu um erro inesperado no sistema: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))
