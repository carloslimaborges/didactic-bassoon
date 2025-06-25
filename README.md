# 📊 Sistema de Análise Fiscal com IA

**Versão 4.0** - Sistema Brasileiro de Análise de Documentos Fiscais

## 🇧🇷 Sobre o Sistema

Sistema inteligente para análise automatizada de documentos fiscais brasileiros, desenvolvido com Inteligência Artificial usando LLaMA e interface web moderna com Streamlit.

### ✨ Principais Funcionalidades

- 📄 **Suporte a múltiplos formatos**: CSV individuais ou arquivos ZIP
- 🤖 **Análise com IA**: Powered by LLaMA via Groq
- 🇧🇷 **Especializado no Brasil**: CFOPs, NCMs, legislação fiscal brasileira
- 📊 **Interface intuitiva**: Dashboard web responsivo
- 💬 **Consultas em linguagem natural**: Pergunte em português
- 🔍 **Análise detalhada**: Inconsistências, relatórios, insights

## 🚀 Instalação Rápida

### Instalar uv
Siga as instruções em https://docs.astral.sh/uv/getting-started/installation/

Abra um terminal e execute o seguinte:
```bash
uv sync
source .venv/bin/activate
python
```

Dentro do comando do python que abrir, execute os comandos abaixo (demora um pouco até finalizar e aparecer >>> novamente apos o primeiro comando):
```python
import pandas as pd
import streamlit as st
```

Feito isso, digite CTRL+D para sair do comando e execute o sistema de fato:
```bash
# Executar sistema
streamlit run agente_fiscal.py
```

Uma janela do navegador deve abrir, e nela pode-se entrar com a chave grok obtida em [console.groq.com](https://console.groq.com) e enviar os arquivos e realizar as analises.

## 📋 Requisitos do Sistema

### Obrigatórios
- **Python 3.8+** (recomendado 3.10+)
- **Chave API Groq** (gratuita em [console.groq.com](https://console.groq.com))
- **4GB RAM** mínimo (8GB recomendado)
- **Conexão com internet** para IA

### Sistema Operacional
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Ubuntu 18.04+
- ✅ Outras distribuições Linux

## 🔧 Configuração

### 1. Obter Chave API Groq
1. Acesse [console.groq.com](https://console.groq.com)
2. Crie uma conta gratuita
3. Gere sua API Key
4. Mantenha a chave segura

### 2. Configurar Variáveis de Ambiente (Opcional)
Crie um arquivo `.env`:
```env
GROQ_API_KEY=sua_chave_api_aqui
STREAMLIT_SERVER_PORT=8501
```

## 🎯 Como Usar

### 1. Iniciar o Sistema
```bash
streamlit run agente_fiscal.py
```

### 2. Acessar Interface
- Abra seu navegador em: http://localhost:8501
- Configure sua API Key na barra lateral
- Carregue seus dados fiscais

### 3. Carregar Dados
- **CSV individual**: Notas fiscais, documentos fiscais
- **Arquivo ZIP**: Múltiplos CSVs organizados
- **Dados exemplo**: Para teste inicial

### 4. Fazer Análises
- Use perguntas sugeridas ou
- Digite consultas personalizadas em português
- Aguarde a análise da IA

## 📊 Exemplos de Consultas

```
✅ "Qual o valor total das notas fiscais?"
✅ "Quais os principais clientes por volume?"
✅ "Quantos CFOPs diferentes temos?"
✅ "Identifique inconsistências nos documentos"
✅ "Calcule o ICMS total recolhido"
✅ "Análise de vendas por região"
```

## 🗂️ Estrutura de Dados Suportada

### Colunas Recomendadas
- `numero_nf`: Número da nota fiscal
- `cfop`: Código Fiscal de Operações
- `valor_total`: Valor total do documento
- `valor_icms`: Valor do ICMS
- `produto`: Descrição do produto
- `cliente`: Nome do cliente
- `uf`: Unidade federativa
- `data_emissao`: Data de emissão
- `ncm`: Nomenclatura Comum do Mercosul

### Formato de Arquivo
```csv
numero_nf,cfop,valor_total,valor_icms,produto,cliente,uf,data_emissao
000001,5102,1500.00,270.00,Mouse Óptico,Tech Solutions,SP,2024-01-15
000002,5102,2300.50,414.09,Teclado,Informática Brasil,RJ,2024-01-16
```

## 🔐 Segurança e Privacidade

- ✅ **Dados locais**: Processamento na sua máquina
- ✅ **API segura**: Comunicação criptografada com Groq
- ✅ **Sem armazenamento**: Dados não são salvos remotamente
- ✅ **Código aberto**: Transparência total

## 🐛 Solução de Problemas

### Erro de API Key
- Verifique se a chave está correta
- Confirme se há créditos na conta Groq
- Teste a chave em [console.groq.com](https://console.groq.com)

### Performance Lenta
- Use modelos menores (llama3-8b-8192)
- Reduza o tamanho dos dados
- Feche outras aplicações

## 📞 Suporte

### Documentação
- [Wiki do Projeto](https://github.com/usuario/analise-fiscal-ia/wiki)
- [FAQ](https://github.com/usuario/analise-fiscal-ia/wiki/FAQ)

### Comunidade
- [Issues no GitHub](https://github.com/usuario/analise-fiscal-ia/issues)
- [Discussões](https://github.com/usuario/analise-fiscal-ia/discussions)

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 🎖️ Créditos

- **Streamlit**: Framework web
- **LangChain**: Orquestração de IA
- **Groq**: Inferência rápida de LLMs
- **LLaMA**: Modelo de linguagem
- **Pandas**: Manipulação de dados

---

**Sistema de Análise Fiscal IA** - Automatizando a análise fiscal brasileira com inteligência artificial

*Desenvolvido com ❤️ para contadores e gestores brasileiros*