
import streamlit as st
import pandas as pd
import zipfile
import os
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import Tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings

# Configurações iniciais
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class FerramentasAvancadas:
    """Classe que encapsula ferramentas avançadas de análise de dados"""

    def __init__(self, df, llm):
        self.df = df
        self.llm = llm

    def informacoes_dataframe(self, pergunta: str = "Informações gerais do DataFrame") -> str:
        """Fornece informações detalhadas sobre o DataFrame"""
        shape = self.df.shape
        columns = self.df.dtypes
        nulos = self.df.isnull().sum()
        nans_str = self.df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
        duplicados = self.df.duplicated().sum()

        template_resposta = PromptTemplate(
            template="""
            Você é um analista de dados especializado. Com base na pergunta "{pergunta}", 
            forneça um resumo detalhado sobre o DataFrame:

            ================= INFORMAÇÕES DO DATAFRAME =================
            Dimensões: {shape}
            Colunas e tipos: {columns}
            Valores nulos: {nulos}
            Strings 'nan': {nans_str}
            Linhas duplicadas: {duplicados}
            ============================================================

            Forneça:
            1. Resumo executivo das dimensões
            2. Análise de cada coluna (tipo e significado)
            3. Avaliação da qualidade dos dados
            4. Sugestões de análises possíveis
            5. Recomendações de tratamento
            """,
            input_variables=["pergunta", "shape", "columns", "nulos", "nans_str", "duplicados"]
        )

        cadeia = template_resposta | self.llm | StrOutputParser()
        return cadeia.invoke({
            "pergunta": pergunta,
            "shape": shape,
            "columns": columns,
            "nulos": nulos,
            "nans_str": nans_str,
            "duplicados": duplicados
        })

    def resumo_estatistico(self, pergunta: str = "Resumo estatístico do DataFrame") -> str:
        """Gera resumo estatístico completo do DataFrame"""
        estatisticas = self.df.describe(include='number').transpose().to_string()

        template_resposta = PromptTemplate(
            template="""
            Baseado na pergunta "{pergunta}", analise as estatísticas descritivas:

            ================= ESTATÍSTICAS DESCRITIVAS =================
            {resumo}
            ============================================================

            Forneça análise detalhada incluindo:
            1. Visão geral das variáveis numéricas
            2. Identificação de possíveis outliers
            3. Padrões e tendências observados
            4. Recomendações para próximos passos
            """,
            input_variables=["pergunta", "resumo"]
        )

        cadeia = template_resposta | self.llm | StrOutputParser()
        return cadeia.invoke({"pergunta": pergunta, "resumo": estatisticas})

    def gerar_grafico(self, pergunta: str) -> str:
        """Gera gráficos baseados na pergunta do usuário"""
        colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in self.df.dtypes.items()])
        amostra_dados = self.df.head(3).to_dict(orient='records')

        template_resposta = PromptTemplate(
            template="""
            Gere código Python para plotar um gráfico baseado em: "{pergunta}"

            Colunas disponíveis:
            {colunas}

            Amostra dos dados:
            {amostra}

            IMPORTANTE:
            - Use matplotlib.pyplot (plt) e seaborn (sns)
            - Configure sns.set_theme()
            - Use figsize=(10, 6)
            - Adicione título e labels apropriados
            - Use sns.despine()
            - Termine com plt.show()

            Retorne APENAS o código Python:
            """,
            input_variables=["pergunta", "colunas", "amostra"]
        )

        cadeia = template_resposta | self.llm | StrOutputParser()
        codigo_bruto = cadeia.invoke({
            "pergunta": pergunta,
            "colunas": colunas_info,
            "amostra": amostra_dados
        })

        # Limpa e executa o código
        codigo_limpo = codigo_bruto.replace("```python", "").replace("```", "").strip()

        try:
            exec_globals = {'df': self.df, 'plt': plt, 'sns': sns, 'pd': pd}
            exec(codigo_limpo, exec_globals)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close()
            return "Gráfico gerado com sucesso!"
        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"

def descompactar_e_ler_csvs(arquivo_zip):
    """Descompacta arquivo ZIP e lê todos os CSVs encontrados"""
    try:
        dataframes = []
        nomes_arquivos = []

        with zipfile.ZipFile(arquivo_zip, 'r') as z:
            arquivos_csv = [f for f in z.namelist() if f.lower().endswith('.csv')]

            if not arquivos_csv:
                st.error("Nenhum arquivo CSV encontrado no ZIP.")
                return None, None

            for nome_arquivo in arquivos_csv:
                with z.open(nome_arquivo) as csv_file:
                    conteudo_bytes = csv_file.read()

                    # Tenta diferentes encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            conteudo_str = conteudo_bytes.decode(encoding)
                            df = pd.read_csv(StringIO(conteudo_str))
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        st.error(f"Erro de encoding no arquivo {nome_arquivo}")
                        continue

                    dataframes.append(df)
                    nomes_arquivos.append(nome_arquivo)

        return dataframes, nomes_arquivos

    except Exception as e:
        st.error(f"Erro ao processar ZIP: {e}")
        return None, None

def verificar_ollama():
    """Verifica se o Ollama está instalado e rodando"""
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    st.set_page_config(
        page_title="🤖 Agente Avançado de Análise de Dados - Gemma 3", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 Agente Avançado de Análise de Dados")
    st.markdown("### Análise inteligente de dados CSV com Gemma 3 local + Ferramentas Sofisticadas")

    # Sidebar com configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Seleção do modelo
        model_name = st.selectbox(
            "Modelo Gemma 3",
            ("gemma3:4b", "gemma3:1b", "gemma3:12b", "gemma3:27b"),
            help="Certifique-se de ter o modelo baixado: ollama pull modelo"
        )

        # Configurações avançadas
        st.subheader("Configurações Avançadas")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_iterations = st.number_input("Max Iterations", 1, 20, 5)

        # Status do Ollama
        st.subheader("🔍 Status do Sistema")
        ollama_status = verificar_ollama()
        if ollama_status:
            st.success("✅ Ollama está rodando")
        else:
            st.error("❌ Ollama não detectado")
            st.info("Execute: ollama serve")

    # Verificação inicial
    if not ollama_status:
        st.error("🚨 Ollama não está rodando! Execute 'ollama serve' no terminal.")
        st.stop()

    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "📁 Upload de arquivo ZIP contendo CSVs",
        type="zip",
        help="Selecione um arquivo ZIP com um ou mais arquivos CSV"
    )

    if uploaded_file is not None:
        with st.spinner('📊 Processando arquivos...'):
            dfs, nomes = descompactar_e_ler_csvs(uploaded_file)

        if dfs:
            st.success(f"✅ {len(dfs)} arquivo(s) carregado(s) com sucesso!")

            # Exibir informações dos arquivos
            with st.expander("📋 Detalhes dos Arquivos Carregados"):
                for nome, df in zip(nomes, dfs):
                    st.write(f"**📄 {nome}**")
                    st.write(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
                    st.dataframe(df.head(3), use_container_width=True)
                    st.divider()

            # Interface principal de análise
            st.header("🧠 Análise Inteligente com Gemma 3")

            # Sugestões de perguntas
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Informações Gerais", use_container_width=True):
                    st.session_state.pergunta_sugerida = "Mostre as informações gerais do dataset"

            # with col2:
            #     if st.button("📈 Resumo Estatístico", use_container_width=True):
            #         st.session_state.pergunta_sugerida = "Faça um resumo estatístico completo dos dados"

            # with col3:
            #     if st.button("📊 Criar Gráfico", use_container_width=True):
            #         st.session_state.pergunta_sugerida = "Crie um gráfico interessante dos dados"

            # Campo de pergunta
            pergunta_inicial = getattr(st.session_state, 'pergunta_sugerida', '')
            pergunta = st.text_area(
                "💬 Faça sua pergunta sobre os dados:",
                value=pergunta_inicial,
                placeholder="Ex: Qual é a correlação entre as variáveis? Mostre a distribuição da variável vendas...",
                height=100
            )

            # Botões de ação
            col1, col2 = st.columns([3, 1])
            with col1:
                analisar = st.button("🚀 Analisar com IA", type="primary", use_container_width=True)
            with col2:
                if st.button("🧹 Limpar", use_container_width=True):
                    st.session_state.pergunta_sugerida = ''
                    st.rerun()

            if analisar and pergunta:
                with st.spinner(f'🤖 {model_name} processando sua solicitação...'):
                    try:
                        # Configurar LLM
                        llm = Ollama(
                            model=model_name,
                            temperature=temperature,
                            num_gpu=1
                        )

                        # Inicializar ferramentas
                        df_principal = dfs[0]  # Usar o primeiro DataFrame
                        ferramentas = FerramentasAvancadas(df_principal, llm)

                        # Determinar tipo de análise baseado na pergunta
                        pergunta_lower = pergunta.lower()

                        if any(word in pergunta_lower for word in ['informações', 'info', 'estrutura', 'colunas', 'geral']):
                            # Usar ferramenta de informações
                            resultado = ferramentas.informacoes_dataframe(pergunta)
                            st.markdown("## 📋 Informações do Dataset")
                            st.markdown(resultado)

                        elif any(word in pergunta_lower for word in ['estatística', 'resumo', 'descritiva', 'média', 'estatístico']):
                            # Usar ferramenta de resumo estatístico
                            resultado = ferramentas.resumo_estatistico(pergunta)
                            st.markdown("## 📊 Análise Estatística")
                            st.markdown(resultado)

                        elif any(word in pergunta_lower for word in ['gráfico', 'plot', 'visualiz', 'chart', 'distribuição']):
                            # Usar ferramenta de gráficos
                            st.markdown("## 📈 Visualização dos Dados")
                            resultado = ferramentas.gerar_grafico(pergunta)
                            if "sucesso" not in resultado:
                                st.error(resultado)

                        else:
                            # Usar agente pandas padrão para consultas mais complexas
                            agente = create_pandas_dataframe_agent(
                                llm,
                                dfs,
                                verbose=True,
                                allow_dangerous_code=True,  # CORREÇÃO DO ERRO PRINCIPAL
                                max_iterations=max_iterations,
                                max_execution_time=300,
                                agent_executor_kwargs={"handle_parsing_errors": True}
                            )

                            resultado = agente.run(pergunta)
                            st.markdown("## 🤖 Resposta do Agente IA")
                            st.markdown(resultado)

                    except Exception as e:
                        st.error(f"❌ Erro na análise: {str(e)}")
                        if "allow_dangerous_code" in str(e):
                            st.info("💡 Erro de segurança resolvido automaticamente. Tente novamente.")
                        elif "connection" in str(e).lower():
                            st.info("🔧 Verifique se o Ollama está rodando: 'ollama serve'")

            elif analisar and not pergunta:
                st.warning("⚠️ Por favor, digite uma pergunta antes de analisar.")

    else:
        # Instruções quando não há arquivo
        st.info("👆 Faça upload de um arquivo ZIP contendo arquivos CSV para começar a análise.")

        # Seção de ajuda
        with st.expander("📚 Como usar este sistema"):
            st.markdown("""
            ### 🚀 Passo a passo:

            1. **Prepare seus dados**: Coloque um ou mais arquivos CSV em um ZIP
            2. **Faça upload**: Use o botão acima para enviar o arquivo
            3. **Escolha uma análise**: Use os botões de sugestão ou digite sua pergunta
            4. **Analise**: Clique em "Analisar com IA" e aguarde o resultado

            ### 💡 Exemplos de perguntas:
            - "Mostre as informações gerais do dataset"
            - "Faça um resumo estatístico das variáveis numéricas" 
            - "Crie um gráfico de distribuição da coluna vendas"
            - "Qual é a correlação entre preço e quantidade?"
            - "Identifique outliers nos dados"

            ### ⚙️ Configuração do Ollama:
            ```bash
            # Instalar Ollama
            curl -fsSL https://ollama.ai/install.sh | sh

            # Baixar modelo Gemma 3
            ollama pull gemma3:4b

            # Iniciar servidor
            ollama serve
            ```
            """)

if __name__ == "__main__":
    main()
