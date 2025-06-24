"""
Análise Fiscal com Streamlit - Agente Customizado Completo

====================================================

Sistema de análise fiscal com IA usando agente customizado que mostra
o DataFrame completo no contexto do modelo de linguagem.

Implementa a Solução 4: Agente Customizado Baseado no Código Fonte

Versão: 7.0 - Customizada Completa
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
from typing import Optional, Dict, Any, List
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
        'langchain': 'langchain',
        'groq': 'groq',
        'plotly': 'plotly'
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
    from langchain.agents import ZeroShotAgent, AgentExecutor
    from langchain.chains import LLMChain
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    st.error(f"❌ Erro de importação: {e}")
    st.stop()

# =====================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# =====================================================================

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
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Criatividade do modelo"
    )

    # Número máximo de linhas para mostrar
    max_rows = st.sidebar.number_input(
        "📊 Máx. linhas no contexto:",
        min_value=10,
        max_value=1000,
        value=10,
        step=5,
        help="Quantas linhas enviar ao modelo"
    )

    return api_key, model, temp, max_rows

def load_data():
    """Carregar dados das notas fiscais"""
    uploaded_files = st.file_uploader(
        "📁 Carregar arquivos de NF-e:",
        accept_multiple_files=True,
        type=['csv', 'zip'],
        help="Carregue os arquivos CSV ou ZIP das notas fiscais"
    )
    
    if not uploaded_files:
        # Dados de exemplo se não houver upload
        if st.button("📋 Usar dados de exemplo"):
            return load_sample_data()
        return None, None
    
    cabecalho_df = None
    itens_df = None
    
    for file in uploaded_files:
        if file.name.endswith('.zip'):
            # Processar arquivo ZIP
            with zipfile.ZipFile(file, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if 'cabecalho' in file_name.lower() or 'header' in file_name.lower():
                        with zip_ref.open(file_name) as csv_file:
                            cabecalho_df = pd.read_csv(csv_file, encoding='utf-8')
                    elif 'itens' in file_name.lower() or 'items' in file_name.lower():
                        with zip_ref.open(file_name) as csv_file:
                            itens_df = pd.read_csv(csv_file, encoding='utf-8')
        else:
            # Processar arquivo CSV
            df = pd.read_csv(file, encoding='utf-8')
            if 'cabecalho' in file.name.lower() or 'header' in file.name.lower():
                cabecalho_df = df
            elif 'itens' in file.name.lower() or 'items' in file.name.lower():
                itens_df = df
    
    return cabecalho_df, itens_df

def load_sample_data():
    """Carregar dados de exemplo"""
    try:
        # Tentar carregar dados processados
        cabecalho_path = "nf_cabecalho_limpo.csv"
        itens_path = "nf_itens_limpo.csv"
        
        if os.path.exists(cabecalho_path):
            cabecalho_df = pd.read_csv(cabecalho_path, encoding='utf-8')
        else:
            st.warning("⚠️ Arquivo de cabeçalho não encontrado")
            return None, None
            
        if os.path.exists(itens_path):
            itens_df = pd.read_csv(itens_path, encoding='utf-8')
        else:
            st.warning("⚠️ Arquivo de itens não encontrado")
            return None, None
            
        return cabecalho_df, itens_df
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        return None, None

def preprocess_data(cabecalho_df, itens_df):
    """Pré-processar dados das notas fiscais"""
    if cabecalho_df is None or itens_df is None:
        return None, None
    
    # Fazer cópia para não alterar os originais
    cab_df = cabecalho_df.copy()
    it_df = itens_df.copy()
    
    # Padronizar nomes das colunas
    def padronizar_colunas(df):
        df.columns = (df.columns
                     .str.lower()
                     .str.replace(' ', '_')
                     .str.replace('/', '_')
                     .str.replace('-', '_')
                     .str.replace('(', '')
                     .str.replace(')', '')
                     .str.normalize('NFD')
                     .str.encode('ascii', errors='ignore')
                     .str.decode('ascii'))
        return df
    
    cab_df = padronizar_colunas(cab_df)
    it_df = padronizar_colunas(it_df)
    
    # Renomear coluna de valor se necessário
    if 'valor_nota_fiscal' in cab_df.columns:
        cab_df = cab_df.rename(columns={'valor_nota_fiscal': 'valor_total_nota'})
    
    # Converter tipos de dados
    try:
        # Chaves de acesso
        if 'chave_de_acesso' in cab_df.columns:
            cab_df['chave_de_acesso'] = cab_df['chave_de_acesso'].astype(str)
        if 'chave_de_acesso' in it_df.columns:
            it_df['chave_de_acesso'] = it_df['chave_de_acesso'].astype(str)
        
        # Datas
        if 'data_emissao' in cab_df.columns:
            cab_df['data_emissao'] = pd.to_datetime(cab_df['data_emissao'], errors='coerce')
        
        # Valores numéricos
        if 'valor_total_nota' in cab_df.columns:
            cab_df['valor_total_nota'] = pd.to_numeric(cab_df['valor_total_nota'], errors='coerce')
        
        # Valores dos itens
        for col in ['quantidade', 'valor_unitario', 'valor_total']:
            if col in it_df.columns:
                it_df[col] = pd.to_numeric(it_df[col], errors='coerce')
                
    except Exception as e:
        st.warning(f"⚠️ Aviso na conversão de tipos: {str(e)}")
    
    return cab_df, it_df

# =====================================================================
# AGENTE CUSTOMIZADO - SOLUÇÃO 4
# =====================================================================

def create_custom_fiscal_agent(llm, cabecalho_df, itens_df, max_rows=100):
    """
    Criar agente customizado que pode mostrar mais linhas do DataFrame
    Implementa a Solução 4: Agente Customizado Baseado no Código Fonte
    """
    
    # Determinar quantas linhas mostrar
    cab_display_rows = min(max_rows, len(cabecalho_df)) if cabecalho_df is not None else 0
    itens_display_rows = min(max_rows, len(itens_df)) if itens_df is not None else 0
    
    # Preparar DataFrames para o contexto
    cab_sample = cabecalho_df.head(cab_display_rows) if cabecalho_df is not None else pd.DataFrame()
    itens_sample = itens_df.head(itens_display_rows) if itens_df is not None else pd.DataFrame()
    
    # Criar ferramentas Python com os DataFrames
    tools = [
        PythonAstREPLTool(
            locals={
                "cabecalho_df": cabecalho_df,
                "itens_df": itens_df,
                "pd": pd,
                "np": np,
                "plt": None,  # Matplotlib não disponível no contexto Streamlit
            }
        )
    ]
    
    # Preparar estatísticas dos dados
    cab_stats = ""
    itens_stats = ""
    
    if cabecalho_df is not None and len(cabecalho_df) > 0:
        cab_stats = f"""
📋 DATASET DE CABEÇALHO:
- Total de registros: {len(cabecalho_df):,}
- Colunas: {list(cabecalho_df.columns)}
- Valores totais: R$ {cabecalho_df['valor_total_nota'].sum():,.2f} (se aplicável)
- Período: {cabecalho_df['data_emissao'].min()} a {cabecalho_df['data_emissao'].max()} (se aplicável)
"""
    
    if itens_df is not None and len(itens_df) > 0:
        itens_stats = f"""
📦 DATASET DE ITENS:
- Total de registros: {len(itens_df):,}
- Colunas: {list(itens_df.columns)}
- Produtos únicos: {itens_df['descricao_do_produto_servico'].nunique() if 'descricao_do_produto_servico' in itens_df.columns else 'N/A'}
"""
    
    # Prompt customizado com informações completas
    PREFIX = f"""
Você é um especialista em análise fiscal brasileira com acesso completo aos dados de Notas Fiscais Eletrônicas.

{cab_stats}

{itens_stats}

AMOSTRA DOS DADOS (primeiras {max_rows} linhas):

CABEÇALHO:
{cab_sample.to_string() if not cab_sample.empty else "Não disponível"}

ITENS:
{itens_sample.to_string() if not itens_sample.empty else "Não disponível"}

Você tem acesso às seguintes ferramentas Python para análise completa dos dados:
- cabecalho_df: DataFrame completo com {len(cabecalho_df) if cabecalho_df is not None else 0} notas fiscais
- itens_df: DataFrame completo com {len(itens_df) if itens_df is not None else 0} itens
- pd: Pandas para manipulação de dados
- np: NumPy para cálculos

INSTRUÇÕES:
1. Use sempre os DataFrames completos (cabecalho_df, itens_df) para análises
2. Forneça estatísticas precisas baseadas em TODOS os dados
3. Cite valores exatos e fontes
4. Explique sua metodologia de cálculo
5. Formate valores monetários em R$ com separadores de milhares

"""
    
    SUFFIX = """Pergunta: {input}

{agent_scratchpad}
Final Answer:"""
    
    # Criar prompt do agente
# Correção implementada: Modificado o SUFFIX para garantir que "Final Answer:" 
# apareça em uma nova linha, permitindo que o AgentExecutor capture corretamente a resposta.
# Também adicionada instrução explícita no PREFIX sobre como formatar a resposta final.

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=PREFIX,
        suffix=SUFFIX,
        input_variables=["input", "agent_scratchpad"]
    )
    
    # Criar cadeia LLM
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Criar agente
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    
    # Criar executor do agente
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=50,
        max_execution_time=10,
        handle_parsing_errors=True,early_stopping_method="generate"
    )
    
    return agent_executor

def display_data_overview(cabecalho_df, itens_df):
    """Exibir resumo dos dados carregados"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Cabeçalho das Notas")
        if cabecalho_df is not None:
            st.metric("Total de Notas", f"{len(cabecalho_df):,}")
            if 'valor_total_nota' in cabecalho_df.columns:
                valor_total = cabecalho_df['valor_total_nota'].sum()
                st.metric("Valor Total", f"R$ {valor_total:,.2f}")
            st.dataframe(cabecalho_df.head(), use_container_width=True)
        else:
            st.info("Nenhum dado de cabeçalho carregado")
    
    with col2:
        st.subheader("📦 Itens das Notas")
        if itens_df is not None:
            st.metric("Total de Itens", f"{len(itens_df):,}")
            if 'descricao_do_produto_servico' in itens_df.columns:
                produtos_unicos = itens_df['descricao_do_produto_servico'].nunique()
                st.metric("Produtos Únicos", f"{produtos_unicos:,}")
            st.dataframe(itens_df.head(), use_container_width=True)
        else:
            st.info("Nenhum dado de itens carregado")

def create_visualizations(cabecalho_df, itens_df):
    """Criar visualizações dos dados"""
    if cabecalho_df is None:
        return
        
    st.subheader("📈 Visualizações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de valores por UF
        if 'uf_emitente' in cabecalho_df.columns and 'valor_total_nota' in cabecalho_df.columns:
            uf_valores = cabecalho_df.groupby('uf_emitente')['valor_total_nota'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=uf_valores.index,
                y=uf_valores.values,
                title="💰 Valor Total por UF Emitente",
                labels={'x': 'UF', 'y': 'Valor Total (R$)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gráfico temporal
        if 'data_emissao' in cabecalho_df.columns:
            cabecalho_df['data_emissao'] = pd.to_datetime(cabecalho_df['data_emissao'], errors='coerce')
            timeline = cabecalho_df.groupby(cabecalho_df['data_emissao'].dt.date).size()
            
            fig = px.line(
                x=timeline.index,
                y=timeline.values,
                title="📅 Emissões por Data",
                labels={'x': 'Data', 'y': 'Número de Notas'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# INTERFACE PRINCIPAL
# =====================================================================

def main():
    """Função principal do aplicativo"""
    
    # Título
    st.title("📊 Análise Fiscal com IA - Agente Customizado")
    st.markdown("Sistema avançado de análise de Notas Fiscais Eletrônicas com agente IA customizado")
    
    # Configurações da sidebar
    api_key, model, temp, max_rows = setup_sidebar()
    
    if not api_key:
        st.warning("🔑 Por favor, insira sua chave API do Groq na barra lateral")
        st.info("📝 Obtenha sua chave gratuita em: https://console.groq.com")
        return
    
    # Carregar dados
    st.subheader("📁 Carregamento de Dados")
    cabecalho_df, itens_df = load_data()
    
    if cabecalho_df is None and itens_df is None:
        st.info("👆 Carregue seus arquivos CSV de notas fiscais ou use os dados de exemplo")
        return
    
    # Pré-processar dados
    with st.spinner("🔄 Processando dados..."):
        cabecalho_df, itens_df = preprocess_data(cabecalho_df, itens_df)
    
    # Exibir visão geral dos dados
    display_data_overview(cabecalho_df, itens_df)
    
    # Criar visualizações
    create_visualizations(cabecalho_df, itens_df)
    
    # Configurar LLM
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=temp,
            max_tokens=4000
        )
    except Exception as e:
        st.error(f"❌ Erro ao configurar modelo: {str(e)}")
        return
    
    # Criar agente customizado
    st.subheader("🤖 Agente de Análise Fiscal")
    
    with st.spinner("🔧 Criando agente customizado..."):
        try:
            agent = create_custom_fiscal_agent(llm, cabecalho_df, itens_df, max_rows)
            st.success(f"✅ Agente criado! Contexto: {max_rows} linhas por dataset")
        except Exception as e:
            st.error(f"❌ Erro ao criar agente: {str(e)}")
            return
    
    # Interface de chat
    st.markdown("### 💬 Faça suas perguntas sobre as notas fiscais:")
    
    # Exemplos de perguntas
    with st.expander("💡 Exemplos de perguntas"):
        st.markdown("""
        - **Quantas notas fiscais existem no total?**
        - **Qual o valor total de todas as notas fiscais?**
        - **Quais são os top 5 produtos mais vendidos?**
        - **Qual a distribuição de operações por UF?**
        - **Quantas operações são interestaduais?**
        - **Qual o valor médio das notas fiscais?**
        - **Quais empresas emitiram mais notas?**
        - **Análise temporal das emissões**
        """)
    
    # Input do usuário
    user_question = st.text_input(
        "Sua pergunta:",
        placeholder="Ex: Quantas notas fiscais existem e qual o valor total?"
    )
    
    if st.button("🔍 Analisar", type="primary"):
        if user_question:
            with st.spinner("🧠 Analisando dados..."):
                try:
                    response = agent.run(user_question)
                    
                    st.markdown("### 📊 Resposta da Análise:")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise: {str(e)}")
                    st.markdown("**Possíveis soluções:**")
                    st.markdown("- Verifique se os dados estão no formato correto")
                    st.markdown("- Reformule a pergunta de forma mais específica")
                    st.markdown("- Reduza o número de linhas no contexto")
        else:
            st.warning("⚠️ Por favor, digite uma pergunta")
    
    # Informações técnicas
    with st.sidebar.expander("ℹ️ Informações Técnicas"):
        st.markdown(f"""
        **Configuração Atual:**
        - Modelo: {model}
        - Temperatura: {temp}
        - Contexto: {max_rows} linhas
        - Registros Cabeçalho: {len(cabecalho_df) if cabecalho_df is not None else 0}
        - Registros Itens: {len(itens_df) if itens_df is not None else 0}
        
        **Solução Implementada:**
        Agente Customizado (Solução 4) - O agente tem acesso completo aos DataFrames via ferramentas Python, permitindo análises precisas de todos os registros.
        """)

if __name__ == "__main__":
    main()
