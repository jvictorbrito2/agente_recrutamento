import streamlit as st
import pandas as pd
import duckdb
import json
import google.generativeai as genai
import time
import re

# --- Configuração da Página ---
st.set_page_config(
    page_title="Decision - Assistente de Recrutamento",
    page_icon="✨",
    layout="wide"
)

# --- Configuração da API do Gemini ---
with st.sidebar:
    st.header("Configuração Essencial")
    google_api_key = st.text_input("Insira sua Chave de API do Google Gemini", type="password", help="Sua chave é necessária para os agentes de IA funcionarem.")
    if google_api_key:
        try:
            genai.configure(api_key=google_api_key)
            st.success("API do Gemini configurada com sucesso!")
        except Exception as e:
            st.error(f"Erro ao configurar a API: {e}")
    st.markdown("---")
    st.info("Este é um MVP. Os arquivos JSON devem estar no mesmo diretório que o `app.py`.")


# --- Funções de Carregamento de Dados ---
@st.cache_data
def carregar_vagas():
    try:
        with open('vagas.json', 'r', encoding='utf-8') as f:
            vagas_data = json.load(f)
        vagas_lista = []
        for codigo, dados in vagas_data.items():
            vaga_info = {
                'codigo_vaga': codigo,
                'titulo_vaga': dados.get('informacoes_basicas', {}).get('titulo_vaga', 'N/A'),
                'cliente': dados.get('informacoes_basicas', {}).get('cliente', 'N/A'),
                'perfil_vaga': dados.get('perfil_vaga', {})
            }
            vagas_lista.append(vaga_info)
        return pd.DataFrame(vagas_lista)
    except FileNotFoundError:
        st.error("Arquivo 'vagas.json' não encontrado.")
        return pd.DataFrame()

@st.cache_data
def carregar_prospects():
    try:
        with open('prospects.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Arquivo 'prospects.json' não encontrado.")
        return {}

def buscar_detalhes_candidato(codigos_candidatos):
    if not codigos_candidatos:
        return pd.DataFrame()
    codigos_str = ", ".join([f"'{c}'" for c in codigos_candidatos])
    query = f"""
    SELECT 
        codigo_candidato,
        informacoes_profissionais ->> 'conhecimentos' AS conhecimentos,
        informacoes_profissionais ->> 'area_de_atuacao' AS area_atuacao,
        informacoes_academicas ->> 'nivel_ingles' AS nivel_ingles,
        informacoes_profissionais ->> 'nivel_profissional' as nivel_profissional,
        cv
    FROM read_json_auto('applicants.json')
    WHERE codigo_candidato IN ({codigos_str})
    """
    try:
        with duckdb.connect(database=':memory:', read_only=False) as con:
            return con.execute(query).fetchdf()
    except Exception:
        with open('applicants.json', 'w') as f:
            json.dump({}, f)
        return pd.DataFrame()

# --- Funções do Agente 1 (Matching Híbrido) ---
@st.cache_data
def analisar_competencias_vaga(competencias_texto):
    """Usa a IA para extrair e categorizar competências da vaga."""
    if not google_api_key: return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analise a descrição de competências de uma vaga de TI e extraia as informações em formato JSON.
        
        Descrição: "{competencias_texto}"

        Seu objetivo é identificar:
        1.  `obrigatorias`: Uma lista das 5 competências técnicas mais essenciais (ex: linguagens, frameworks).
        2.  `desejaveis`: Uma lista de outras competências técnicas ou comportamentais mencionadas.
        3.  `sinonimos`: Para cada competência obrigatória, gere uma lista de 2-3 sinônimos ou tecnologias relacionadas que um bom candidato poderia mencionar.

        Retorne APENAS o objeto JSON, sem nenhum outro texto ou formatação. Exemplo de saída:
        {{
            "obrigatorias": ["Python", "Django", "API REST", "PostgreSQL", "AWS"],
            "desejaveis": ["React", "Docker", "Metodologias Ágeis"],
            "sinonimos": {{
                "Python": ["Pandas", "Numpy", "Flask"],
                "AWS": ["EC2", "S3", "Lambda"]
            }}
        }}
        """
        response = model.generate_content(prompt)
        # Limpa a resposta para garantir que é um JSON válido
        json_response = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_response)
    except Exception as e:
        st.error(f"Erro na IA ao analisar competências: {e}")
        return None

def calcular_score_hibrido(candidato_texto, competencias_analisadas):
    """Calcula o score de um candidato com base na análise híbrida."""
    if not competencias_analisadas: return 0
    
    score = 0
    candidato_texto = candidato_texto.lower()
    
    # Pontuação para competências obrigatórias e seus sinônimos
    for comp, sinonimos in competencias_analisadas.get('sinonimos', {}).items():
        if comp.lower() in candidato_texto:
            score += 10 # Ponto principal
        for s in sinonimos:
            if s.lower() in candidato_texto:
                score += 5 # Ponto por sinônimo
                break # Conta apenas uma vez por grupo de sinônimos
    
    # Pontuação para competências desejáveis
    for comp in competencias_analisadas.get('desejaveis', []):
        if comp.lower() in candidato_texto:
            score += 3

    return score


# --- Funções do Agente 2 (Entrevista e Análise) ---
def gerar_relatorio_final(vaga, candidato, historico_chat):
    """Gera o relatório final da entrevista usando a IA."""
    if not google_api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""
    Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.

    **Vaga:** {vaga.get('titulo_vaga', 'N/A')} (Cliente: {vaga.get('cliente', 'N/A')})
    **Competências:** {vaga.get('perfil_vaga', {}).get('competencia_tecnicas_e_comportamentais', 'N/A')}
    **Candidato:** {candidato.get('nome', 'N/A')} (CV: {candidato.get('cv', 'N/A')})
    **Transcrição da Entrevista:**\n{historico_chat}

    **Sua Tarefa:** Gere um relatório estruturado avaliando o candidato nos 3 pilares: Análise Técnica, Fit Cultural e Engajamento/Motivação.

    **Formato (use exatamente este markdown):**
    ### Relatório Final de Entrevista - {candidato.get('nome', 'N/A')}
    **1. Score Geral:** (Nota de 0 a 10)
    **2. Análise de Pontos Fortes:** (bullet points)
    **3. Análise de Pontos de Atenção:** (bullet points)
    **4. Recomendação Final:** ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado", com breve justificativa)
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar o relatório: {e}"

def gerar_analise_comparativa(vaga, relatorios):
    """Gera uma análise comparativa de todos os finalistas."""
    if not google_api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""
    Você é o Diretor de Recrutamento da Decision. Sua tarefa é analisar os relatórios de todos os finalistas para a vaga abaixo e eleger o(s) candidato(s) perfeito(s).

    **Vaga:** {vaga.get('titulo_vaga', 'N/A')} (Cliente: {vaga.get('cliente', 'N/A')})

    **Relatórios dos Finalistas:**
    ---
    {relatorios}
    ---

    **Sua Tarefa:**
    1. Crie um ranking dos candidatos, do mais recomendado ao menos.
    2. Escreva um parecer final, justificando sua escolha pelo(s) candidato(s) "perfeito(s)". Destaque como o candidato escolhido se sobressai em relação aos outros nos pilares de avaliação (técnico, cultural, engajamento).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar a análise comparativa: {e}"

# --- Interface Principal ---
st.title("✨ Assistente de Recrutamento da Decision")
st.markdown("Bem-vindo ao seu assistente de IA para otimizar o processo de seleção.")

# Carrega os dados
df_vagas = carregar_vagas()
prospects_data = carregar_prospects()

# Inicializa o session_state
if 'candidatos_para_entrevista' not in st.session_state: st.session_state.candidatos_para_entrevista = []
if 'vaga_selecionada' not in st.session_state: st.session_state.vaga_selecionada = {}
if "messages" not in st.session_state: st.session_state.messages = {}
if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}

tab1, tab2, tab3 = st.tabs(["Agente 1: Matching Inteligente", "Agente 2: Entrevistas", "Análise Final Comparativa"])

with tab1:
    st.header("Análise e Matching de Vagas")
    if not google_api_key:
        st.warning("Por favor, insira sua chave de API do Google na barra lateral para usar o Agente 1.")
    elif not df_vagas.empty and prospects_data:
        
        # --- Barra de Busca para Vagas ---
        termo_busca = st.text_input("Buscar vaga por título, cliente ou código:", placeholder="Ex: Python, Morris, 5185")
        
        df_vagas_filtrado = df_vagas
        if termo_busca:
            termo_busca = termo_busca.lower()
            df_vagas_filtrado = df_vagas[
                df_vagas['titulo_vaga'].str.lower().str.contains(termo_busca) |
                df_vagas['cliente'].str.lower().str.contains(termo_busca) |
                df_vagas['codigo_vaga'].str.contains(termo_busca)
            ]

        if df_vagas_filtrado.empty:
            st.warning("Nenhuma vaga encontrada com o termo de busca.")
        else:
            opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for index, row in df_vagas_filtrado.iterrows()}
            codigo_vaga_selecionada = st.selectbox("Selecione a vaga:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x], key="select_vaga")

            if codigo_vaga_selecionada:
                vaga_selecionada_data = df_vagas[df_vagas['codigo_vaga'] == codigo_vaga_selecionada].iloc[0]
                perfil_vaga = vaga_selecionada_data['perfil_vaga']
                competencias_texto = perfil_vaga.get('competencia_tecnicas_e_comportamentais', 'N/A')

                with st.expander("Ver detalhes da vaga"):
                    st.write(f"**Competências:** {competencias_texto}")

                if st.button("Analisar Candidatos com IA", type="primary"):
                    with st.spinner("Agente 1 está analisando as competências da vaga..."):
                        competencias_analisadas = analisar_competencias_vaga(competencias_texto)
                    
                    if competencias_analisadas:
                        st.success("Competências analisadas pela IA!")
                        with st.expander("Ver competências extraídas pela IA"):
                            st.json(competencias_analisadas)

                        candidatos_prospect = prospects_data.get(codigo_vaga_selecionada, {}).get('prospects', [])
                        if candidatos_prospect:
                            with st.spinner("Buscando e pontuando candidatos..."):
                                df_prospects = pd.DataFrame(candidatos_prospect)
                                codigos_lista = df_prospects['codigo'].tolist()
                                df_detalhes = buscar_detalhes_candidato(codigos_lista)

                                if not df_detalhes.empty:
                                    df_prospects = df_prospects.rename(columns={'codigo': 'codigo_candidato'})
                                    df_resultado = pd.merge(df_prospects, df_detalhes, on='codigo_candidato', how='left').fillna('')
                                    
                                    df_resultado['texto_completo'] = df_resultado['conhecimentos'] + " " + df_resultado['cv']
                                    df_resultado['score'] = df_resultado['texto_completo'].apply(lambda x: calcular_score_hibrido(x, competencias_analisadas))
                                    df_resultado = df_resultado.sort_values(by='score', ascending=False).head(10)

                                    st.subheader("Top 10 Candidatos Recomendados")
                                    df_resultado['selecionar'] = False
                                    df_editado = st.data_editor(
                                        df_resultado[['selecionar', 'nome', 'score', 'conhecimentos']],
                                        column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar", default=False)},
                                        hide_index=True, use_container_width=True
                                    )
                                    
                                    if st.button("Confirmar Seleção para Entrevista"):
                                        candidatos_selecionados = df_editado[df_editado['selecionar']]
                                        if not candidatos_selecionados.empty:
                                            st.session_state.candidatos_para_entrevista = candidatos_selecionados.to_dict('records')
                                            st.session_state.vaga_selecionada = vaga_selecionada_data.to_dict()
                                            st.session_state.relatorios_finais[codigo_vaga_selecionada] = {} # Limpa relatórios antigos
                                            st.success(f"{len(candidatos_selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                                            time.sleep(2)
                                            st.rerun()
                                        else:
                                            st.warning("Nenhum candidato selecionado.")
                        else:
                            st.warning("Nenhum prospect encontrado para esta vaga.")
                    else:
                        st.error("Não foi possível analisar as competências da vaga. Verifique a API Key e a descrição da vaga.")

with tab2:
    st.header("Condução das Entrevistas")
    if not st.session_state.candidatos_para_entrevista:
        st.info("Nenhum candidato selecionado. Volte para a aba 'Matching' para selecionar.")
    else:
        vaga_atual = st.session_state.vaga_selecionada
        st.subheader(f"Vaga: {vaga_atual.get('titulo_vaga', 'N/A')}")
        
        nomes_candidatos = {c['codigo_candidato']: c['nome'] for c in st.session_state.candidatos_para_entrevista}
        id_candidato = st.selectbox("Selecione o candidato para entrevistar:", options=list(nomes_candidatos.keys()), format_func=lambda x: nomes_candidatos[x])
        candidato_atual = [c for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato][0]

        # Inicializa o estado do chat para o candidato
        if id_candidato not in st.session_state.messages:
            st.session_state.messages[id_candidato] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{candidato_atual['nome']}**."}]
        
        # Exibe o chat
        chat_container = st.container(height=400)
        for message in st.session_state.messages[id_candidato]:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input do recrutador
        if prompt := st.chat_input("Digite a resposta do candidato..."):
            st.session_state.messages[id_candidato].append({"role": "user", "content": prompt})
            with st.spinner("Agente 2 está pensando..."):
                historico_formatado = "\n".join([f"{'Recrutador' if m['role'] == 'user' else 'IA'}: {m['content']}" for m in st.session_state.messages[id_candidato]])
                prompt_ia = f"Continue a entrevista de forma natural, com base no histórico. Faça a próxima pergunta para avaliar o candidato para a vaga de {vaga_atual['titulo_vaga']}.\nHistórico:\n{historico_formatado}\n\nSua próxima pergunta:"
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt_ia)
                st.session_state.messages[id_candidato].append({"role": "assistant", "content": response.text})
            st.rerun()

        # Botão para gerar relatório individual
        codigo_vaga_atual = vaga_atual.get('codigo_vaga')
        if st.button(f"🏁 Finalizar Entrevista e Gerar Relatório para {candidato_atual['nome']}"):
            with st.spinner("Analisando entrevista e gerando relatório..."):
                historico_final = "\n".join([f"{'Candidato' if m['role'] == 'user' else 'Entrevistador'}: {m['content']}" for m in st.session_state.messages[id_candidato]])
                relatorio = gerar_relatorio_final(vaga_atual, candidato_atual, historico_final)
                
                if codigo_vaga_atual not in st.session_state.relatorios_finais:
                    st.session_state.relatorios_finais[codigo_vaga_atual] = {}
                st.session_state.relatorios_finais[codigo_vaga_atual][id_candidato] = relatorio
                
                st.success("Relatório gerado! Verifique a aba 'Análise Final Comparativa'.")
                st.markdown(relatorio)

with tab3:
    st.header("Análise Final e Decisão")
    if not st.session_state.vaga_selecionada:
        st.info("Nenhuma vaga em processo de análise. Comece pela aba 'Matching'.")
    else:
        codigo_vaga_atual = st.session_state.vaga_selecionada.get('codigo_vaga')
        relatorios_vaga_atual = st.session_state.relatorios_finais.get(codigo_vaga_atual, {})

        if not relatorios_vaga_atual:
            st.info("Nenhum relatório de entrevista foi gerado para esta vaga ainda. Finalize as entrevistas na aba 'Entrevistas'.")
        else:
            st.subheader(f"Finalistas para a vaga: {st.session_state.vaga_selecionada.get('titulo_vaga')}")
            for id_candidato, relatorio in relatorios_vaga_atual.items():
                nome_candidato = [c['nome'] for c in st.session_state.candidatos_para_entrevista if c['codigo_candidato'] == id_candidato][0]
                with st.expander(f"Ver relatório de {nome_candidato}"):
                    st.markdown(relatorio)
            
            if len(relatorios_vaga_atual) >= 2:
                if st.button("Gerar Análise Comparativa Final com IA", type="primary"):
                    with st.spinner("IA está analisando todos os finalistas para eleger o melhor..."):
                        todos_relatorios = "\n\n---\n\n".join(relatorios_vaga_atual.values())
                        analise_final = gerar_analise_comparativa(st.session_state.vaga_selecionada, todos_relatorios)
                        st.markdown("---")
                        st.subheader("Parecer Final do Assistente de IA")
                        st.markdown(analise_final)
            else:
                st.info("Você precisa finalizar pelo menos duas entrevistas para gerar uma análise comparativa.")
