import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool

from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {'Groq': 
                        {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                         'chat': ChatGroq},
                  'OpenAI': 
                        {'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
                         'chat': ChatOpenAI}}

MEMORIA = ConversationBufferMemory()

PERFIS_AGENTES = {
    'Especialista em IA': {
        'descricao': '''Você é um agente especialista em IA, focado em análise de documentação, 
        versões e especificações técnicas. Forneça informações técnicas precisas e análise 
        de compatibilidade de versões.''',
        'ferramentas': ['busca_documentacao', 'verificacao_versao']
    },
    'Consultor Criativo': {
        'descricao': '''Você é um agente consultor criativo que avalia ideias e fornece 
        feedback construtivo. Foque em inovação, viabilidade e possíveis melhorias.''',
        'ferramentas': ['avaliacao_ideias', 'analise_mercado']
    },
    'Pesquisador Web': {
        'descricao': '''Você é um agente pesquisador especializado em encontrar e verificar 
        informações de fontes da internet. Foque em precisão e cite fontes confiáveis.''',
        'ferramentas': ['busca_web', 'verificacao_fatos']
    }
}


def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    agentes_selecionados = st.session_state.get('agentes_selecionados', ['Especialista em IA'])
    
    instrucoes_agentes = "\n\n".join([
        f"Quando atuando como {agente}:\n{PERFIS_AGENTES[agente]['descricao']}"
        for agente in agentes_selecionados
    ])

    system_message = f'''Você é o GascIA, um assistente multifuncional que pode alternar entre diferentes papéis.
    Você tem acesso às informações de um documento {tipo_arquivo} e pode realizar buscas na internet:

    ####
    {documento}
    ####

    {instrucoes_agentes}

    Para realizar buscas na internet, use a ferramenta de busca disponível.
    Quando usar informações da internet, cite a fonte.

    Com base na pergunta do usuário, adote o papel mais apropriado dentre os agentes selecionados.
    Sempre indique qual papel de agente você está usando para responder.

    Sempre que houver $ na sua saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o GascAI!'''

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    
    if 'Pesquisador Web' in agentes_selecionados:
        chat.tools = [FERRAMENTAS['busca_web']]
    
    chain = template | chat
    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖Bem-vindo ao GascAI', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carrege o GascAI')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o GascAI')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
            }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos', 'Seleção de Agentes'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['.txt'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key para o provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))
        st.session_state[f'api_key_{provedor}'] = api_key
    with tabs[2]:
        agentes_selecionados = st.multiselect(
            'Selecione os agentes para sua consulta',
            options=list(PERFIS_AGENTES.keys()),
            default=['Especialista em IA']
        )
        st.session_state['agentes_selecionados'] = agentes_selecionados
    
    if st.button('Inicializar GascAI', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()