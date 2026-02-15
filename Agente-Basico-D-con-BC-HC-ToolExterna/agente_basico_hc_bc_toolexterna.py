"""
Agente IA Completo: Base de Conocimiento + Internet + Histórico
- Tool 1: Base de Conocimiento (RAG con Supabase)
- Tool 2: Búsqueda en Internet (Tavily)
- Histórico: Guarda conversaciones en PostgreSQL

Autor: Ing. Kevin Inofuente Colque - DataPath
"""
import os
import sys
import uuid
from urllib.parse import quote_plus
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Agregar el directorio raíz al path para importar tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

# Importar tools desde la carpeta tools/
from tools.Base_de_conocimiento import buscar_datapath
from tools.Busqueda_internet import buscar_internet
from tools.Hora_y_fecha import obtener_fecha_hora

# ============================================
# 1. CONFIGURACIÓN DE BASE DE DATOS (Histórico)
# ============================================
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

if not all([DB_USER, DB_PASSWORD, DB_HOST]):
    raise ValueError(
        "❌ Faltan variables de base de datos en .env\n"
        "Requeridas: DB_USER, DB_PASSWORD, DB_HOST"
    )

DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"🔌 Conectando como: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ============================================
# 2. LISTA DE TOOLS DISPONIBLES
# ============================================
tools = [
    buscar_datapath,   # Base de conocimiento DATAPATH
    buscar_internet,   # Búsqueda en internet (Tavily)
    obtener_fecha_hora
]

# ============================================
# 3. CONFIGURACIÓN DEL MODELO CON TOOLS
# ============================================
chat = init_chat_model("gpt-4.1", temperature=0.7)
chat_con_tools = chat.bind_tools(tools)

# ============================================
# 4. PROMPT DEL AGENTE
# ============================================
system_prompt = """Eres DataBot, un asistente de IA de DATAPATH con acceso a internet.

Tu objetivo es ayudar a los usuarios respondiendo sus preguntas usando las herramientas disponibles.

HERRAMIENTAS DISPONIBLES:
1. buscar_datapath: Para información sobre DATAPATH (programas, cursos, precios, docentes)
2. buscar_internet: Para información actualizada de internet (noticias, eventos, datos actuales)

INSTRUCCIONES:
- Para preguntas sobre DATAPATH → USA buscar_datapath
- Para preguntas sobre eventos actuales, noticias, o información general → USA buscar_internet
- Para saludos, agradecimientos o conversación general → Responde directamente SIN herramientas
- Puedes usar AMBAS herramientas si la pregunta lo requiere
- Recuerdas toda la conversación gracias a tu memoria persistente
- Responde siempre en español de manera clara y amigable

EJEMPLOS:
- "Hola" → Responde directamente
- "¿Qué cursos tienen?" → Usa buscar_datapath
- "¿Qué pasó hoy en las noticias?" → Usa buscar_internet
- "¿Cómo se compara su curso de IA con las tendencias actuales?" → Usa AMBAS tools"""

# ============================================
# 5. CREAR TABLA DE HISTORIAL
# ============================================
def crear_tabla_historial():
    try:
        sync_connection = psycopg.connect(DATABASE_URL)
        PostgresChatMessageHistory.create_tables(sync_connection, "chat_history_whatsapp")
        sync_connection.close()
    except Exception as e:
        print(f"⚠️ Nota sobre tabla: {e}")

crear_tabla_historial()

# ============================================
# 6. HISTÓRICO DE CONVERSACIÓN
# ============================================
def get_session_history(session_id: str) -> PostgresChatMessageHistory:
    sync_connection = psycopg.connect(DATABASE_URL)
    return PostgresChatMessageHistory(
        "chat_history_whatsapp",
        session_id,
        sync_connection=sync_connection
    )

# ============================================
# 7. FUNCIÓN DE CHAT CON AGENTE + TOOLS
# ============================================
def chat_con_agente(mensaje_usuario: str, session_id: str) -> str:
    """
    Ejecuta el agente con tools y memoria.
    El agente decide si usar herramientas o responder directamente.
    """
    # Obtener historial
    history = get_session_history(session_id)
    mensajes_previos = history.messages
    
    # Construir mensajes para el modelo
    messages = [{"role": "system", "content": system_prompt}]
    
    # Agregar historial
    for msg in mensajes_previos:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    
    # Agregar mensaje actual
    messages.append({"role": "user", "content": mensaje_usuario})
    
    # Invocar modelo con tools
    response = chat_con_tools.invoke(messages)
    
    # Procesar tool calls si existen
    if response.tool_calls:
        # Ejecutar cada tool
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Buscar y ejecutar la tool
            for t in tools:
                if t.name == tool_name:
                    result = t.invoke(tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "result": result
                    })
                    break
        
        # Agregar respuesta del modelo con tool calls y resultados
        messages.append(response)
        for tr in tool_results:
            messages.append(ToolMessage(
                content=tr["result"],
                tool_call_id=tr["tool_call_id"]
            ))
        
        # Segunda llamada para obtener respuesta final
        final_response = chat_con_tools.invoke(messages)
        respuesta_final = final_response.content
    else:
        # Sin tool calls, respuesta directa
        respuesta_final = response.content
    
    # Guardar en historial
    history.add_user_message(mensaje_usuario)
    history.add_ai_message(respuesta_final)
    
    return respuesta_final


# ============================================
# 8. LOOP DE CONVERSACIÓN
# ============================================
def main():
    print("=" * 60)
    print("🤖 DataBot - Agente COMPLETO (BC + Internet + Memoria)")
    print("=" * 60)
    print("🔧 Tools disponibles:")
    for t in tools:
        print(f"   - {t.name}")
    print("💾 Historial: PostgreSQL")
    
    # Menú de sesión
    print("\nOpciones de sesión:")
    print("  1. Nueva conversación")
    print("  2. Continuar sesión existente (pegar UUID)")
    
    opcion = input("\nElige (1/2): ").strip()
    
    if opcion == "2":
        session_id = input("Pega el UUID de la sesión: ").strip()
        try:
            uuid.UUID(session_id)
        except ValueError:
            print("⚠️ UUID inválido. Creando nueva sesión...")
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())
    
    print(f"\n📝 Session ID: {session_id}")
    print("   (Guarda este ID para continuar después)")
    print("✅ El agente puede buscar en DATAPATH y en INTERNET")
    print("Escribe 'salir' para volver al menú.\n")
    
    while True:
        usuario = input("Tú: ").strip()
        
        if usuario.lower() in ['salir', 'exit', 'quit']:
            print(f"\n💾 Tu sesión está guardada.")
            print(f"   UUID: {session_id}")
            print("👋 ¡Hasta luego!")
            break
        
        if not usuario:
            continue
        
        try:
            respuesta = chat_con_agente(usuario, session_id)
            print(f"\n🤖 DataBot: {respuesta}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
