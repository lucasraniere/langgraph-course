import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


load_dotenv()
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

# Definir nós de agentes
def agente_de_voo(state: MessagesState):
    """Agente especializado em busca de vôos"""
    print("-- Agente de voo --")
    sys_msg = SystemMessage(content='Você é um especialista em reservas de vôos. Foque em achar e explorar APENAS opções de vôos, preços e itinerários. Recomende os vôos mais próximos, também escolha o melhor aeroporto. Responda apenas sobre vôos.')
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


def agente_de_hospedagem(state: MessagesState):
    """Agente especializado em busca de hospedagem"""
    print("-- Agente de Hospedagem --")
    sys_msg = SystemMessage(content='Você é um especialista em hotéis. Foque em achar e explorar APENAS opções de hospedagem, preços e acomodações. Dê exemplos com preços e localidades variadas. Responda apenas sobre hospedagem.')
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


def agente_de_restaurantes(state: MessagesState):
    """Agente especializado em recomendação de restaurantes"""
    print("-- Agente de Restaurantes --")
    sys_msg = SystemMessage(content='Você é um especialista em restaurantes. Foque em achar e explorar APENAS opções de restaurantes, preços e cozinhas. Responda apenas sobre restaurantes.')
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


def agente_de_atracoes(state: MessagesState):
    """Agente especializado em atrações turísticas"""
    print("-- Agente de Atrações --")
    sys_msg = SystemMessage(content='Você é um especialista em atrações turísticas. Foque em achar e explorar APENAS opções de atrações, atividades e experiências. Responda apenas sobre atividades e pontos turísticos.')
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


def agente_coordenador(state: MessagesState):
    """Agente coordenador que decide qual agente usar"""
    sys_msg = SystemMessage(content="""Você é um coordenador de viagens. Analise o pedido do usuário e decida qual especialista solicitar pesquisa:
    - Para buscas referentes a vôos, responda com "DIRECIONAR_PARA: VOO"
    - Para buscas referentes a hospedagem, responda com "DIRECIONAR_PARA: HOSPEDAGEM"
    - Para buscas referentes a restaurantes, responda com "DIRECIONAR_PARA: RESTAURANTE
    - Para buscas referentes a atrações, responda com "DIRECIONAR_PARA: ATRACAO"
    - Se uma requisição precisar de mais de um espcialista, liste eles em ordem: "DIRECIONAR_PARA: [ESPECIALISTA_1] [ESPECIALISTA_2] ...s [ESPECIALISTA_N]".
    Você também deve informar o que o usuário está solicitando.
    - Caso a pergunta não esteja relacionada a viagens, retorne END.
    """)
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


def agente_agregador(state: MessagesState):
    """Agente que agrega as respostas de todos os agentes"""
    responses = [msg.content for msg in state["messages"]]
    prompt = [
        SystemMessage(content="Você deve agregar e resumir todas as respostas que você receber dos agentes em um output."),
        HumanMessage(content=f"Agora, agregue as seguintes respostas encontradas: {responses}")
    ]
    result = llm.invoke(prompt)
    return {"messages": result}

# Construindo o grafo
builder = StateGraph(MessagesState)

# Adicionando nós
builder.add_node("coordenador", agente_coordenador)
builder.add_node("agente_voo", agente_de_voo)
builder.add_node("agente_hospedagem", agente_de_hospedagem)
builder.add_node("agente_restaurante", agente_de_restaurantes)
builder.add_node("agente_atracao", agente_de_atracoes)
builder.add_node("agregador", agente_agregador)

# Adicionando arestas
builder.add_edge(START, "coordenador")

# Adicionando arestas condicionais baseando-se na escolha do coordenador
def route_to_agent(state):
    """Direciona para o agente apropriado com base na escolha do coordenador"""
    last_message = state["messages"][-1]
    agents_list = []
    if not isinstance(last_message, AIMessage):
        agents_list.append("coordenador")


    content = last_message.content
    if "VOO" in content:
        agents_list.append("agente_voo")
    if "HOSPEDAGEM" in content:
         agents_list.append("agente_hospedagem")
    if "RESTAURANTE" in content:
        agents_list.append("agente_restaurante")
    if "ATRACAO" in content:
        agents_list.append("agente_atracao")
    else:
        agents_list.append(END)
    return agents_list


builder.add_conditional_edges(
    "coordenador",
    route_to_agent,
    {
        "agente_voo": "agente_voo",
        "agente_hospedagem": "agente_hospedagem",
        "agente_restaurante": "agente_restaurante",
        "agente_atracao": "agente_atracao",
        "coordenador": "coordenador",
        END:END
    }
)

builder.add_edge("agente_voo", "agregador")
builder.add_edge("agente_hospedagem", "agregador")
builder.add_edge("agente_restaurante", "agregador")
builder.add_edge("agente_atracao", "agregador")

builder.add_edge("agregador", END)

# Compilar grafo
grafo_viagem = builder.compile()