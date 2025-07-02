import streamlit as st
import pandas as pd
import duckdb
import json
import google.generativeai as genai
import time
import re
import os
import requests

# --- Configuração da Página ---
st.set_page_config(
    page_title="Decision - Assistente de Recrutamento",
    page_icon="✨",
    layout="wide"
)

# --- URLs para os arquivos no Hugging Face ---
# !!! IMPORTANTE: Substitua pelas suas URLs !!!
APPLICANTS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/applicants.json"
VAGAS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/vagas.json"
PROSPECTS_JSON_URL = "https://huggingface.co/datasets/jvictorbrito/agente_recrutamento/resolve/main/prospects.json"
NDJSON_FILENAME = "applicants_nd.json"


# --- Funções de Preparação de Dados ---
def preparar_dados_candidatos(url, original_filename, ndjson_filename):
    """
    Garante que o arquivo de dados dos candidatos esteja disponível no formato otimizado (NDJSON).
    Se o arquivo otimizado não existir, baixa o original e o converte.
    """
    if os.path.exists(ndjson_filename):
        return True

    # Baixa o arquivo original se ele também não existir
    if not os.path.exists(original_filename):
        st.info(f"Arquivo de dados '{original_filename}' não encontrado. Baixando do repositório...")
        try:
            with st.spinner(f"Baixando {original_filename}... (Isso pode levar alguns minutos na primeira vez)"):
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(original_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.success(f"Arquivo '{original_filename}' baixado com sucesso!")
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao baixar o arquivo '{original_filename}': {e}. Verifique a URL.")
            st.stop()

    # Converte o JSON original para NDJSON
    st.info(f"Primeiro uso: Convertendo '{original_filename}' para um formato otimizado...")
    with st.spinner("Isso pode levar um momento, mas só acontecerá uma vez."):
        try:
            with open(original_filename, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
            
            with open(ndjson_filename, 'w', encoding='utf-8') as f_out:
                for codigo, candidato_data in data.items():
                    # Adiciona o código do candidato dentro do próprio objeto
                    candidato_data['codigo_candidato'] = codigo
                    json.dump(candidato_data, f_out)
                    f_out.write('\n')
            st.success("Arquivo de dados otimizado com sucesso!")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao converter o arquivo JSON: {e}")
            st.stop()

def baixar_arquivo_se_nao_existir(url, nome_arquivo):
    """Baixa arquivos JSON menores se não existirem."""
    if not os.path.exists(nome_arquivo):
        st.info(f"Baixando arquivo de configuração '{nome_arquivo}'...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(nome_arquivo, 'wb') as f:
                f.write(response.content)
            st.success(f"'{nome_arquivo}' baixado.")
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao baixar '{nome_arquivo}': {e}.")
            st.stop()

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
    # CORREÇÃO: Mensagem da barra lateral simplificada.
    st.info("Assistente de Recrutamento IA - MVP")

# --- Download e Preparação dos Dados ---
baixar_arquivo_se_nao_existir(VAGAS_JSON_URL, "vagas.json")
baixar_arquivo_se_nao_existir(PROSPECTS_JSON_URL, "prospects.json")
preparar_dados_candidatos(APPLICANTS_JSON_URL, "applicants.json", NDJSON_FILENAME)


# --- Funções de Carregamento de Dados ---
@st.cache_data
def carregar_vagas():
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

@st.cache_data
def carregar_prospects():
    with open('prospects.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def buscar_detalhes_candidato(codigos_candidatos):
    if not codigos_candidatos:
        return pd.DataFrame()
    codigos_str = ", ".join([f"'{c}'" for c in codigos_candidatos])
    query = f"""
    SELECT 
        codigo_candidato,
        informacoes_profissionais ->> 'conhecimentos' AS conhecimentos,
        informacoes_profissionais ->> 'area_de_atuacao' AS area_atuacao,
        formacao_e_idiomas ->> 'nivel_ingles' AS nivel_ingles,
        informacoes_profissionais ->> 'nivel_profissional' as nivel_profissional,
        (cv_pt || ' ' || cv_en) AS cv
    FROM read_json_auto('{NDJSON_FILENAME}')
    WHERE codigo_candidato IN ({codigos_str})
    """
    try:
        with duckdb.connect(database=':memory:', read_only=False) as con:
            return con.execute(query).fetchdf()
    except Exception as e:
        st.error(f"Erro ao consultar o arquivo de candidatos com DuckDB: {e}")
        return pd.DataFrame()

# --- Funções do Agente 1 (Matching Híbrido) ---
@st.cache_data
def analisar_competencias_vaga(competencias_texto):
    if not google_api_key: return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analise a descrição de competências de uma vaga de TI e extraia as informações em formato JSON.
        Descrição: "{competencias_texto}"
        Seu objetivo é identificar:
        1.  `obrigatorias`: Uma lista das 5 competências técnicas mais essenciais.
        2.  `desejaveis`: Uma lista de outras competências.
        3.  `sinonimos`: Para cada competência obrigatória, gere uma lista de 2-3 sinônimos ou tecnologias relacionadas.
        Retorne APENAS o objeto JSON.
        """
        response = model.generate_content(prompt)
        json_response = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_response)
    except Exception as e:
        st.error(f"Erro na IA ao analisar competências: {e}")
        return None

def calcular_score_hibrido(candidato_texto, competencias_analisadas):
    if not competencias_analisadas: return 0
    score = 0
    candidato_texto = candidato_texto.lower()
    for comp, sinonimos in competencias_analisadas.get('sinonimos', {}).items():
        if comp.lower() in candidato_texto: score += 10
        for s in sinonimos:
            if s.lower() in candidato_texto:
                score += 5
                break
    for comp in competencias_analisadas.get('desejaveis', []):
        if comp.lower() in candidato_texto: score += 3
    return score

# --- Funções do Agente 2 (Entrevista e Análise) ---
def gerar_relatorio_final(vaga, candidato, historico_chat):
    if not google_api_key: return "Erro: Chave de API do Google não configurada."
    prompt = f"""
    Você é um especialista em recrutamento da Decision. Analise a transcrição de uma entrevista e gere um relatório final.
    **Vaga:** {vaga.get('titulo_vaga', 'N/A')}
    **Candidato:** {candidato.get('nome', 'N/A')}
    **Transcrição:**\n{historico_chat}
    **Sua Tarefa:** Gere um relatório estruturado.
    **Formato (markdown):**
    ### Relatório Final de Entrevista - {candidato.get('nome', 'N/A')}
    **1. Score Geral:** (0 a 10)
    **2. Pontos Fortes:** (bullet points)
    **3. Pontos de Atenção:** (bullet points)
    **4. Recomendação Final:** ("Fit Perfeito", "Recomendado com Ressalvas" ou "Não Recomendado", com justificativa)
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Ocorreu um erro ao gerar o relatório: {e}"

def gerar_analise_comparativa(vaga, relatorios):
    if not google_api_key: return "Erro: Chave de API do Google não configurada."
    # CORREÇÃO: O prompt agora instrui a IA a agir como um recrutador da Decision
    # recomendando um candidato para a empresa CLIENTE.
    cliente = vaga.get('cliente', 'empresa contratante')
    prompt = f"""
    Você é um Diretor de Recrutamento da **Decision**. Sua tarefa é criar um parecer final para apresentar ao seu cliente, a empresa **'{cliente}'**.
    Analise os relatórios de entrevista dos finalistas para a vaga de **{vaga.get('titulo_vaga', 'N/A')}**.

    **Relatórios dos Finalistas:**
    ---
    {relatorios}
    ---

    **Sua Tarefa:**
    1.  Crie um ranking dos candidatos, do mais recomendado para o menos.
    2.  Escreva um parecer final direcionado ao cliente ('{cliente}'), justificando por que você, como representante da Decision, recomenda a contratação do candidato ideal para a vaga deles. Enfatize como as habilidades e o perfil do candidato escolhido beneficiarão o cliente.
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
if "question_count" not in st.session_state: st.session_state.question_count = {}
if "relatorios_finais" not in st.session_state: st.session_state.relatorios_finais = {}
if 'analise_vaga_id' not in st.session_state: st.session_state.analise_vaga_id = None
if 'df_analise_resultado' not in st.session_state: st.session_state.df_analise_resultado = None


tab1, tab2, tab3 = st.tabs(["Agente 1: Matching Inteligente", "Agente 2: Entrevistas", "Análise Final Comparativa"])

with tab1:
    st.header("Análise e Matching de Vagas")
    if not google_api_key:
        st.warning("Por favor, insira sua chave de API do Google na barra lateral para usar o Agente 1.")
    elif not df_vagas.empty and prospects_data:
        termo_busca = st.text_input("Buscar vaga por título, cliente ou código:", placeholder="Ex: Python, Morris, 5185")
        df_vagas_filtrado = df_vagas
        if termo_busca:
            termo_busca = termo_busca.lower()
            df_vagas_filtrado = df_vagas[df_vagas['titulo_vaga'].str.lower().str.contains(termo_busca) | df_vagas['cliente'].str.lower().str.contains(termo_busca) | df_vagas['codigo_vaga'].str.contains(termo_busca)]
        
        if df_vagas_filtrado.empty:
            st.warning("Nenhuma vaga encontrada com o termo de busca.")
        else:
            opcoes_vagas = {row['codigo_vaga']: f"{row['titulo_vaga']} (Cliente: {row['cliente']})" for index, row in df_vagas_filtrado.iterrows()}
            codigo_vaga_selecionada = st.selectbox("Selecione a vaga:", options=list(opcoes_vagas.keys()), format_func=lambda x: opcoes_vagas[x], key="select_vaga")

            if st.session_state.analise_vaga_id != codigo_vaga_selecionada:
                st.session_state.df_analise_resultado = None
                st.session_state.analise_vaga_id = codigo_vaga_selecionada

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
                                    st.session_state.df_analise_resultado = df_resultado
                                else:
                                    st.error("Não foi possível buscar detalhes dos candidatos.")
                        else:
                            st.warning("Nenhum prospect encontrado para esta vaga.")
                    else:
                        st.error("Não foi possível analisar as competências da vaga. Verifique a API Key e a descrição da vaga.")

                if st.session_state.df_analise_resultado is not None and not st.session_state.df_analise_resultado.empty:
                    st.subheader("Top 10 Candidatos Recomendados")
                    df_para_editar = st.session_state.df_analise_resultado.copy()
                    df_para_editar['selecionar'] = False
                    
                    df_editado = st.data_editor(
                        df_para_editar[['selecionar', 'nome', 'score', 'conhecimentos']],
                        column_config={"selecionar": st.column_config.CheckboxColumn("Selecionar para Entrevista", default=False)},
                        hide_index=True,
                        use_container_width=True,
                        key=f"editor_{st.session_state.analise_vaga_id}"
                    )
                    
                    if st.button("Confirmar Seleção para Entrevista"):
                        candidatos_selecionados = df_editado[df_editado['selecionar']]
                        if not candidatos_selecionados.empty:
                            df_selecionados_completo = pd.merge(candidatos_selecionados, st.session_state.df_analise_resultado, on=['nome', 'score', 'conhecimentos'])
                            
                            st.session_state.candidatos_para_entrevista = df_selecionados_completo.to_dict('records')
                            st.session_state.vaga_selecionada = vaga_selecionada_data.to_dict()
                            st.session_state.relatorios_finais[codigo_vaga_selecionada] = {}
                            st.session_state.df_analise_resultado = None
                            
                            st.success(f"{len(candidatos_selecionados)} candidato(s) movido(s) para a aba de entrevistas!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.warning("Nenhum candidato selecionado.")

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
        
        if id_candidato not in st.session_state.messages:
            st.session_state.messages[id_candidato] = [{"role": "assistant", "content": f"Olá! Sou o assistente de IA. Pronto para iniciar a entrevista com **{candidato_atual['nome']}**. Para começar, pode me dizer 'Olá' ou 'Podemos começar'."}]
            st.session_state.question_count[id_candidato] = 0

        chat_container = st.container(height=400)
        for message in st.session_state.messages[id_candidato]:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Digite a resposta do candidato..."):
            st.session_state.messages[id_candidato].append({"role": "user", "content": prompt})
            with st.spinner("Agente 2 está pensando..."):
                historico_formatado = "\n".join([f"{'Recrutador (resposta do candidato)' if m['role'] == 'user' else 'IA (sua pergunta)'}: {m['content']}" for m in st.session_state.messages[id_candidato]])
                
                question_count = st.session_state.question_count.get(id_candidato, 0)
                
                prompt_ia = f"""
                Você é um entrevistador de IA da Decision. Sua tarefa é conduzir uma entrevista concisa e eficaz, fazendo UMA PERGUNTA DE CADA VEZ.

                **Regras da Entrevista:**
                1.  Você está na pergunta **{question_count + 1}** de um total de 5 a 7 perguntas.
                2.  Analise o histórico da conversa e as informações do candidato para formular a **próxima pergunta lógica e relevante**.
                3.  **NÃO FAÇA MAIS DE UMA PERGUNTA.** Apenas a próxima.
                4.  Se você já fez 6 ou 7 perguntas, sua próxima resposta deve ser para **finalizar a entrevista**, agradecendo o candidato e perguntando se ele tem alguma dúvida.

                **Contexto da Vaga:**
                - **Título:** {vaga_atual['titulo_vaga']}
                - **Competências:** {vaga_atual['perfil_vaga'].get('competencia_tecnicas_e_comportamentais')}
                
                **Contexto do Candidato:**
                - **Nome:** {candidato_atual.get('nome')}
                - **Conhecimentos/CV:** {candidato_atual.get('conhecimentos')} | {candidato_atual.get('cv')}

                **Histórico da Conversa até agora:**
                {historico_formatado}

                **Sua próxima ação (formule a PRÓXIMA E ÚNICA pergunta ou finalize a entrevista):**
                """
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt_ia)
                st.session_state.messages[id_candidato].append({"role": "assistant", "content": response.text})
                st.session_state.question_count[id_candidato] = question_count + 1
            st.rerun()

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
