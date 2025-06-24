"""
Análise Fiscal com Streamlit - Versão Final Corrigida
====================================================
Sistema de análise fiscal com IA - Sem conflitos de contexto
Versão: 4.0 - Definitivamente Corrigida
"""

import streamlit as st
import pandas as pd
import os
import zipfile
from typing import Optional, Dict, Any, List

# Configuração da página deve vir PRIMEIRO
st.set_page_config(
    page_title="📊 Análise Fiscal IA - Brasil",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Verificar dependências necessárias"""
    required_packages = {
        'langchain_groq': 'langchain-groq',
        'langchain_experimental': 'langchain-experimental', 
        'groq': 'groq'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        st.error(f"❌ Dependências faltando: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")
        st.stop()

# Verificar dependências
check_dependencies()

# Importações após verificação
try:
    from langchain_groq import ChatGroq
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
except ImportError as e:
    st.error(f"❌ Erro de importação: {e}")
    st.stop()

def setup_sidebar():
    """Configurar barra lateral"""
    st.sidebar.title("⚙️ Configurações")
    
    # API Key
    api_key = st.sidebar.text_input(
        "🔑 Chave API Groq:",
        type="password",
        help="Obtenha em console.groq.com"
    )
    
    # Modelo
    model = st.sidebar.selectbox(
        "🤖 Modelo:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="Modelo de linguagem"
    )
    
    # Temperatura
    temp = st.sidebar.slider(
        "🌡️ Temperatura:",
        0.0, 1.0, 0.1, 0.1,
        help="Controla criatividade"
    )
    
    # Configurações
    st.sidebar.subheader("🔧 Avançado")
    verbose = st.sidebar.checkbox("Modo Detalhado", False)
    dangerous = st.sidebar.checkbox("Análises Complexas", True)
    
    return api_key, model, temp, verbose, dangerous

def process_zip_file(uploaded_zip):
    """Processar arquivo ZIP"""
    dataframes = {}
    
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            files = zip_ref.namelist()
            csv_files = [f for f in files if f.lower().endswith('.csv')]
            
            if not csv_files:
                st.warning("⚠️ Nenhum CSV encontrado no ZIP")
                return None
            
            st.success(f"✅ {len(csv_files)} arquivo(s) CSV encontrado(s)")
            
            for csv_file in csv_files:
                try:
                    with zip_ref.open(csv_file) as file:
                        df = pd.read_csv(file, encoding='utf-8')
                        dataframes[csv_file] = df
                        st.info(f"📊 {csv_file}: {len(df)} registros")
                except Exception as e:
                    st.error(f"❌ Erro em {csv_file}: {e}")
            
            return dataframes
            
    except Exception as e:
        st.error(f"❌ Erro no ZIP: {e}")
        return None

def create_sample_data():
    """Dados de exemplo"""
    return pd.DataFrame({
        'numero_nf': ['001', '002', '003', '004'],
        'cfop': ['5102', '5102', '6102', '5403'],
        'valor_total': [1500.00, 2300.50, 1200.00, 3400.75],
        'valor_icms': [270.00, 414.09, 216.00, 612.14],
        'produto': ['Mouse', 'Teclado', 'Monitor', 'Notebook'],
        'cliente': ['Tech A', 'Tech B', 'Tech C', 'Tech D'],
        'uf': ['SP', 'RJ', 'MG', 'SP'],
        'data': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']
    })

def load_data():
    """Carregar dados"""
    st.header("📁 Dados Fiscais")
    
    uploaded = st.file_uploader(
        "📎 Selecione arquivo:",
        type=['csv', 'zip'],
        help="CSV ou ZIP com CSVs"
    )
    
    use_sample = st.button("🎯 Usar Dados Exemplo")
    
    if uploaded:
        if uploaded.name.endswith('.zip'):
            st.info("📦 Processando ZIP...")
            dataframes = process_zip_file(uploaded)
            if dataframes:
                if len(dataframes) == 1:
                    return list(dataframes.values())[0]
                else:
                    selected = st.selectbox("Selecione arquivo:", list(dataframes.keys()))
                    return dataframes[selected]
        else:
            try:
                df = pd.read_csv(uploaded, encoding='utf-8')
                st.success("✅ CSV carregado!")
                return df
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    elif use_sample:
        st.success("✅ Dados exemplo carregados!")
        return create_sample_data()
    
    return None

def init_agent(api_key, model, temp, df, verbose, dangerous):
    """Inicializar agente"""
    if not api_key:
        return None, "API key necessária"
    
    try:
        llm = ChatGroq(
            model=model,
            temperature=temp,
            groq_api_key=api_key
        )
        
        prompt = """
        Você é especialista em análise fiscal brasileira.
        - Responda em português
        - Use R$ para valores
        - Explique CFOPs e NCMs
        - Seja preciso com cálculos
        """
        
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="tool-calling",
            verbose=verbose,
            allow_dangerous_code=dangerous,
            prefix=prompt
        )
        
        return agent, None
        
    except Exception as e:
        return None, str(e)

def main():
    """Função principal"""
    st.title("📊 Análise Fiscal Inteligente")
    st.markdown("### 🇧🇷 Sistema Brasileiro de Análise Fiscal")
    
    # Sidebar
    api_key, model, temp, verbose, dangerous = setup_sidebar()
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        df = load_data()
        
        if df is not None:
            st.subheader("👀 Prévia")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("📊 Resumo")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Registros", len(df))
            with col_b:
                if 'valor_total' in df.columns:
                    total = df['valor_total'].sum()
                    st.metric("Total", f"R$ {total:,.2f}")
    
    with col2:
        st.subheader("🤖 Assistente IA")
        
        if df is not None:
            if not api_key:
                st.warning("⚠️ Digite a chave API na barra lateral")
                return
            
            # Inicializar agente
            if 'agent' not in st.session_state:
                with st.spinner("🔄 Inicializando IA..."):
                    agent, error = init_agent(api_key, model, temp, df, verbose, dangerous)
                    if agent:
                        st.session_state.agent = agent
                        st.success("✅ IA pronta!")
                    else:
                        st.error(f"❌ Erro: {error}")
            
            # Interface de perguntas
            if 'agent' in st.session_state:
                
                # Perguntas sugeridas
                st.markdown("### 💡 Perguntas Sugeridas")
                questions = [
                    "Qual o valor total das notas fiscais?",
                    "Quais os principais clientes?",
                    "Quantos CFOPs diferentes temos?",
                    "Qual produto vendeu mais?"
                ]
                
                for i, q in enumerate(questions):
                    if st.button(f"❓ {q}", key=f"q_{i}"):
                        st.session_state.selected_question = q
                
                # Input personalizado
                question = st.text_area(
                    "💬 Sua pergunta:",
                    value=st.session_state.get('selected_question', ''),
                    height=100
                )
                
                if st.button("🚀 Analisar", type="primary"):
                    if question.strip():
                        with st.spinner("🔍 Analisando..."):
                            try:
                                response = st.session_state.agent.run(question)
                                st.success("✅ Análise concluída!")
                                st.markdown("### 📋 Resultado")
                                st.markdown(f"**Pergunta:** {question}")
                                st.markdown(f"**Resposta:** {response}")
                            except Exception as e:
                                st.error(f"❌ Erro: {e}")
                    else:
                        st.warning("⚠️ Digite uma pergunta")
        else:
            st.info("ℹ️ Carregue dados para começar")

# Execução - VERSÃO CORRIGIDA
if __name__ == "__main__":
    main()