# app.py
#
# Streamlit verze Business Model Canvas Navigátoru.
# Pro spuštění je nutné mít nastavený GOOGLE_API_KEY v tajných klíčích (Secrets) Streamlitu.

import streamlit as st
import os
import json
import google.generativeai as genai

# ==============================================================================
# --- KONFIGURAČNÍ SEKCE ---
# Parametry jsou zachovány přesně podle původního Colab notebooku.
# ==============================================================================

# Prioritizovaný seznam modelů (s vrácenou verzí 2.5 flash)
PRIORITY_MODEL_STEMS = [
    "gemini-2.5-flash-preview-05-20",  # Vaše preferovaná preview verze
    "gemini-1.5-flash-latest",          # Stabilní, rychlý fallback
    "gemini-1.5-pro-latest",            # Výkonnější fallback
    "gemini-pro",                       # Široce dostupný fallback
]

# Konfigurace generování
GENERATION_CONFIG = {
    "temperature": 1.5,
    "top_p": 0.95,
    "max_output_tokens": 65536,
}

# Bezpečnostní nastavení
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Prompty pro LLM ---

LLM_EXPERT_QUESTION_PLANNER = """
You are an expert strategy consultant and a master of the Alex Osterwalder Business Model Canvas methodology. Your task is to create a structured, comprehensive questioning plan to guide a user through a deep-dive description of their IT Reseller/System Integrator business.

Your output MUST be a valid JSON list of 9 objects, one for each core block of the Business Model Canvas. The order should be logical (Customers/Value -> Operations -> Finances). Each object must have the following four keys:
1. "key": The standard snake_case identifier for the block (e.g., "customer_segments").
2. "question": The main, user-friendly question for the block.
3. "coverage_points": A list of 3-4 critical sub-questions or topics the user MUST consider to provide a complete answer for that block. These should be insightful and cover the nuances of the methodology.
4. "examples": A list of 3-4 short, relevant examples for an IT Reseller/System Integrator.

Example of one object in the list:
{
  "key": "value_propositions",
  "question": "Now, let's detail your Value Propositions. What value do you deliver to your customers?",
  "coverage_points": [
    "Which specific customer problem are you solving or which need are you satisfying?",
    "What bundle of products and services are you offering to each segment?",
    "How does your offering differ from competitors (e.g., is it about performance, price, design, convenience)?",
    "Are you offering something new and disruptive, or improving an existing solution?"
  ],
  "examples": ["Managed Cybersecurity (SOC-as-a-Service)", "Custom Cloud Migration Projects", "Hardware procurement with expert lifecycle advice", "24/7 Premium Technical Support"]
}

Generate ONLY the JSON list and nothing else.
"""

# !!! DŮLEŽITÉ: Nahraďte následující zástupné texty vašimi skutečnými prompty!
LLM_DEEP_ANALYSIS_PERSONA_V2 = """
**[ZÁSTUPNÝ TEXT - NAHRAĎTE SVÝM SKUTEČNÝM PROMPTEM PRO ANALÝZU]**
You are a top-tier business strategist. Analyze the provided Business Model Canvas data for an IT company.
- Identify key strengths and weaknesses in each block.
- Point out potential misalignments between blocks (e.g., value proposition doesn't match customer segment needs).
- Summarize the overall coherence and viability of the business model.
- Present the analysis in a structured, easy-to-read format using Markdown.
"""

LLM_INNOVATION_SUGGESTION_PERSONA_V2 = """
**[ZÁSTUPNÝ TEXT - NAHRAĎTE SVÝM SKUTEČNÝM PROMPTEM PRO NÁVRHY]**
You are a creative innovation consultant specializing in the IT sector. Based on the user's business model data and the strategic analysis, generate actionable, innovative suggestions.
- For each BMC block, provide 1-2 concrete, creative ideas for improvement or new opportunities.
- Ideas should be relevant to an IT Reseller/System Integrator.
- Explain the potential benefit of each suggestion.
- Present the suggestions in a clear, compelling format using Markdown.
"""

# ==============================================================================
# --- POMOCNÉ FUNKCE ---
# ==============================================================================

@st.cache_resource
def initialize_model():
    """Inicializuje a cachuje Gemini model pro celou session."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Chyba při konfiguraci API: Ujistěte se, že máte v Streamlit Secrets nastavený 'GOOGLE_API_KEY'. Detaily: {e}")
        st.stop()

    try:
        model_name_to_use = None
        available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for model_stem in PRIORITY_MODEL_STEMS:
            found_model = next((m for m in available_models if model_stem in m.name and 'vision' not in m.name.lower()), None)
            if found_model:
                model_name_to_use = found_model.name
                break
        
        if not model_name_to_use:
            st.error("Nepodařilo se najít žádný z prioritních Gemini modelů. Zkontrolujte dostupnost modelů ve vaší oblasti a přístup vašeho projektu k preview verzím.")
            st.stop()
            
        model = genai.GenerativeModel(
            model_name=model_name_to_use,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        return model
    except Exception as e:
        st.error(f"Kritická chyba při inicializaci modelu: {e}")
        st.stop()

def ask_gemini_sdk(model, prompt_text: str, temperature: float = None) -> str:
    """Odešle prompt na Gemini a vrátí odpověď."""
    config_overrides = {}
    if temperature is not None:
        config_overrides['temperature'] = float(temperature)

    try:
        response = model.generate_content(prompt_text, generation_config=config_overrides)
        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"AI_ERROR: Váš prompt byl z bezpečnostních důvodů zablokován ({response.prompt_feedback.block_reason.name})."
        else:
            return "AI_ERROR: Model vrátil neúplnou odpověď."
    except Exception as e:
        return f"AI_ERROR: Během volání API nastala chyba: {type(e).__name__}."

def reset_session():
    """Vynuluje session state a spustí aplikaci od začátku."""
    st.session_state.clear()
    st.rerun()

# ==============================================================================
# --- HLAVNÍ LOGIKA APLIKACE ---
# ==============================================================================

# Nastavení stránky
st.set_page_config(page_title="BMC Navigator", page_icon="🚀", layout="wide")
st.title("🚀 BMC Navigator")
st.markdown("Váš AI byznys kouč pro tvorbu a inovaci Business Model Canvas.")

# Inicializace session state
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'welcome'
    st.session_state.question_plan = []
    st.session_state.current_question_index = 0
    st.session_state.bmc_data = {}
    st.session_state.analysis_result = ""
    st.session_state.suggestions_result = ""

# Inicializace modelu
model = initialize_model()

# --- Fáze 1: Úvod a generování plánu ---
if st.session_state.app_stage == 'welcome':
    st.info("Jsem tu, abych vám pomohl zmapovat a inovovat váš IT byznys model. Společně projdeme všech 9 bloků Business Model Canvas.")
    if st.button("Jdeme na to!"):
        with st.spinner("Připravuji pro vás personalizovaný plán dotazování..."):
            plan = ask_gemini_sdk(model, LLM_EXPERT_QUESTION_PLANNER, temperature=0.2)
            if "AI_ERROR" in plan:
                st.error(f"Nepodařilo se vytvořit plán: {plan}")
            else:
                try:
                    cleaned_json_text = plan.strip().lstrip("```json").rstrip("```").strip()
                    st.session_state.question_plan = json.loads(cleaned_json_text)
                    st.session_state.app_stage = 'questioning'
                    st.rerun()
                except (json.JSONDecodeError, ValueError) as e:
                    st.error(f"Chyba při zpracování plánu od AI: {e}")

# --- Fáze 2: Dotazování ---
elif st.session_state.app_stage == 'questioning':
    idx = st.session_state.current_question_index
    plan = st.session_state.question_plan
    
    if idx < len(plan):
        q_config = plan[idx]
        st.progress((idx + 1) / len(plan))
        st.subheader(f"Oblast {idx + 1}/{len(plan)}: {q_config.get('key', 'Neznámý blok').replace('_', ' ').title()}")
        
        st.markdown(f"**{q_config.get('question', '')}**")

        with st.expander("Body k zamyšlení a příklady"):
            st.markdown("###### Pro komplexní odpověď zvažte:")
            for point in q_config.get('coverage_points', []):
                st.markdown(f"- {point}")
            st.markdown("---")
            st.markdown(f"**Příklady:** *{', '.join(q_config.get('examples', []))}*")

        answer = st.text_area("Vaše odpověď:", key=f"answer_{idx}", height=200)

        col1, col2, col3 = st.columns([1,1,5])
        with col1:
            if st.button("Další otázka", type="primary"):
                if len(answer.strip()) < 25:
                    st.warning("Odpověď je velmi stručná. Zkuste prosím přidat více detailů pro lepší analýzu.")
                else:
                    st.session_state.bmc_data[q_config.get('key')] = answer.strip()
                    st.session_state.current_question_index += 1
                    st.rerun()
        with col2:
             if st.button("Přeskočit"):
                st.session_state.bmc_data[q_config.get('key')] = "Skipped"
                st.session_state.current_question_index += 1
                st.rerun()
    else:
        st.success("Skvělá práce! Zmapovali jsme celý váš byznys model.")
        st.session_state.app_stage = 'analysis'
        st.rerun()

# --- Fáze 3 & 4: Analýza a Návrhy ---
elif st.session_state.app_stage == 'analysis':
    with st.spinner("Provádím hloubkovou strategickou analýzu vašich odpovědí..."):
        bmc_data_string = "\n".join([f"- {key}: {value}" for key, value in st.session_state.bmc_data.items() if value != "Skipped"])
        analysis_prompt = f"{LLM_DEEP_ANALYSIS_PERSONA_V2}\n\nHere is the BMC data from the user:\n{bmc_data_string}"
        st.session_state.analysis_result = ask_gemini_sdk(model, analysis_prompt, temperature=0.8)

    with st.spinner("Na základě analýzy generuji inovativní návrhy..."):
        suggestion_prompt = (
            f"{LLM_INNOVATION_SUGGESTION_PERSONA_V2}\n\n"
            f"**User's Business Model Canvas Data:**\n{bmc_data_string}\n\n"
            f"**Strategic Analysis Summary:**\n{st.session_state.analysis_result}\n\n"
            "Now, generate the innovation suggestions based on all the above information."
        )
        st.session_state.suggestions_result = ask_gemini_sdk(model, suggestion_prompt, temperature=1.2)
    
    st.session_state.app_stage = 'done'
    st.rerun()

# --- Fáze 5: Zobrazení výsledků ---
elif st.session_state.app_stage == 'done':
    st.balloons()
    st.header("🎉 Hotovo! Zde jsou výsledky.")

    with st.expander("Vaše zadané informace (Business Model Canvas)", expanded=False):
        for key, value in st.session_state.bmc_data.items():
            st.markdown(f"##### {key.replace('_', ' ').title()}")
            st.markdown(f"> {value}")

    st.markdown("---")
    st.header("📊 Strategická Analýza")
    st.markdown(st.session_state.analysis_result)

    st.markdown("---")
    st.header("💡 Návrhy Inovací")
    st.markdown(st.session_state.suggestions_result)

    st.markdown("---")
    if st.button("Začít znovu"):
        reset_session()
