"""
Agente IA Básico SIN MEMORIA
Solo modelo + prompt - No recuerda conversaciones anteriores

Este es un ejemplo de lo que pasa cuando un agente NO tiene memoria:
- Cada mensaje es independiente
- No recuerda lo que dijiste antes
- No puede mantener contexto de la conversación

Autor: Ing. Kevin Inofuente Colque - DataPath
"""

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# ============================================
# 1. CONFIGURACIÓN DEL MODELO
# ============================================
chat = init_chat_model(
    "gpt-4.1",
    temperature=0.7,
)

# ============================================
# 2. PROMPT SIMPLE (Sin placeholder de historial)
# ============================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un asistente de IA útil y amigable llamado DataBot.
Responde las preguntas del usuario de manera clara y concisa.
Responde siempre en español."""),
    ("human", "{input}")
])

# ============================================
# 3. CREAR LA CHAIN (Solo prompt | modelo)
# ============================================
chain = prompt | chat

# ============================================
# 4. FUNCIÓN SIMPLE SIN MEMORIA
# ============================================
def chat_con_agente(mensaje_usuario: str) -> str:
    """
    Envía un mensaje al agente.
    ⚠️ NO mantiene historial - cada mensaje es independiente.
    """
    respuesta = chain.invoke({"input": mensaje_usuario})
    return respuesta.content

# ============================================
# 5. LOOP DE CONVERSACIÓN
# ============================================
def main():
    print("=" * 50)
    print("🤖 DataBot - Agente SIN MEMORIA")
    print("=" * 50)
    print("⚠️  Este agente NO recuerda mensajes anteriores")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        usuario = input("Tú: ").strip()
        
        if usuario.lower() in ['salir', 'exit', 'quit']:
            print("\n¡Hasta luego! 👋")
            break
        
        if not usuario:
            continue
        
        respuesta = chat_con_agente(usuario)
        print(f"\n🤖 DataBot: {respuesta}\n")
