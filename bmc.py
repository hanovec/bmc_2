# app.py
#
# Finální verze s pokročilými, strukturovanými prompty inspirovanými uživatelskou ukázkou.
# Cílem je generovat výstupy na úrovni skutečného strategického konzultanta.

import streamlit as st
import json
import google.generativeai as genai

# ==============================================================================
# --- KONFIGURAČNÍ SEKCE ---
# ==============================================================================

PRIORITY_MODEL_STEMS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]
GENERATION_CONFIG = {"temperature": 1.5, "top_p": 0.95, "max_output_tokens": 65536}
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- PŘEPRACOVANÉ ŠABLONY PROMPTŮ PRO LLM (V ČEŠTINĚ) ---

LLM_EXPERT_QUESTION_PLANNER_CZ_TEMPLATE = """
Jsi expert na strategické poradenství a mistr metodologie Business Model Canvas. Tvým úkolem je vytvořit strukturovaný a komplexní plán otázek, který provede uživatele hloubkovým popisem jeho byznysu.

KONTEXT OD UŽIVATELE:
---
{business_context}
---

Na základě výše uvedeného kontextu vytvoř plán otázek. Přizpůsob otázky, body k pokrytí a PŘEDEVŠÍM příklady tak, aby co nejlépe odpovídaly byznysu a cílům, které uživatel popsal.

Tvůj výstup MUSÍ být validní JSON list 9 objektů, jeden pro každý blok Business Model Canvas. Pořadí by mělo být logické. Každý objekt musí mít klíče: "key", "question", "coverage_points", "examples". Všechny texty musí být v češtině.

Generuj POUZE JSON list a nic jiného.
"""

LLM_DEEP_ANALYSIS_PERSONA_V2_CZ_TEMPLATE = """
Jsi špičkový byznys stratég a konzultant. Tvým úkolem je vytvořit komplexní strategickou analýzu na základě dat z Business Model Canvas, které ti poskytl klient. Buď strukturovaný, analytický a insightful.

DATA OD KLIENTA:
---
{bmc_data_string}
---

Nyní vytvoř analýzu v následující přesné struktuře s použitím Markdown:

1.  **Executive Summary:** Krátké, výstižné shrnutí klíčových zjištění a strategického napětí v byznys modelu.

2.  **Hloubková analýza:**
    *   **Silné stránky:** Identifikuj 2-3 klíčové silné stránky. Pro každou uveď:
        *   **Nález:** Co je silnou stránkou?
        *   **Důkaz:** Která konkrétní část z dat od klienta to dokazuje? (Cituj!)
        *   **Důsledek:** Jaký pozitivní dopad to má na byznys?
    *   **Slabé stránky:** Identifikuj 2-3 klíčové slabiny. Pro každou uveď:
        *   **Nález:** Co je slabou stránkou?
        *   **Důkaz:** Která konkrétní část z dat od klienta to dokazuje? (Cituj!)
        *   **Důsledek:** Jaké riziko nebo omezení to představuje?
    *   **Příležitosti:** Identifikuj 2-3 největší příležitosti na trhu. Pro každou uveď:
        *   **Nález:** Jaká je příležitost?
        *   **Důkaz:** Která část z dat od klienta (nebo obecná znalost trhu) na ni ukazuje?
        *   **Důsledek:** Jak by firma mohla z této příležitosti profitovat?
    *   **Hrozby:** Identifikuj 2-3 největší hrozby. Pro každou uveď:
        *   **Nález:** Co je hrozbou?
        *   **Důkaz:** Proč je to hrozba pro tento konkrétní byznys model?
        *   **Důsledek:** Jaký negativní dopad by to mohlo mít?

3.  **Klíčová propojení a vzorce:** Popiš 2 nejdůležitější vzorce, které vidíš v byznys modelu. Zaměř se na to, jak se dva nebo více bloků navzájem ovlivňují (např. jak "Vztahy se zákazníky" ovlivňují "Strukturu nákladů").

4.  **Klíčové strategické otázky pro vedení:** Na základě celé analýzy zformuluj 3-4 provokativní, otevřené otázky, které by si mělo vedení firmy položit, aby se posunulo dál.
"""

LLM_INNOVATION_SUGGESTION_PERSONA_V2_CZ_TEMPLATE = """
Jsi kreativní inovační konzultant. Tvým úkolem je na základě poskytnutých dat a strategické analýzy vygenerovat sadu konkrétních, akčních a kreativních návrhů na inovaci byznysu.

DATA OD KLIENTA (BMC):
---
{bmc_data_string}
---
STRATEGICKÁ ANALÝZA:
---
{analysis_summary}
---

Nyní vytvoř návrhy inovací v následující přesné struktuře s použitím Markdown:

Rozděl návrhy do tří kategorií:
-   **Rychlé úspěchy (Quick Wins):** Nápady s nízkou náročností a rychlým dopadem.
-   **Strategické posuny (Strategic Shifts):** Významnější změny, které vyžadují více zdrojů, ale mají velký potenciál.
-   **Experimentální nápady (Moonshots):** Odvážné, dlouhodobé vize, které mohou byznys radikálně proměnit.

Pro každý jednotlivý návrh (minimálně 1-2 v každé kategorii) uveď:
*   **Název návrhu:** Krátký, chytlavý název.
*   **Popis:** Co přesně by firma měla udělat?
*   **Odůvodnění a napojení na analýzu:** Proč je tento návrh relevantní? Na kterou slabinu nebo příležitost z analýzy reaguje?
*   **Dopad na Business Model Canvas:** Které bloky BMC by se změnily a jak?
*   **Akční první kroky (do 30 dnů):** Co jsou 2-3 konkrétní, malé kroky, které lze udělat okamžitě?
*   **Potenciální rizika:** Na co si dát pozor při implementaci?
"""

# ==============================================================================
# --- POMOCNÉ FUNKCE ---
# ==============================================================================

@st.cache_resource
def initialize_model():
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
            st.error("Nepodařilo se najít žádný z prioritních Gemini modelů.")
            st.stop()
        model = genai.GenerativeModel(model_name=model_name_to_use, generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS)
        return model
    except Exception as e:
        st.error(f"Kritická chyba při inicializaci modelu: {e}")
        st.stop()

def ask_gemini_sdk(model, prompt_text: str, temperature: float = None) -> str:
    config_overrides = {}
    if temperature is not None:
        config_overrides['temperature'] = float(temperature)
    try:
        response = model.generate_content(prompt_text, generation_config=config_overrides)
        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"AI_CHYBA: Požadavek byl zablokován ({response.prompt_feedback.block_reason.name})."
        else:
            return "AI_CHYBA: Model vrátil neúplnou odpověď."
    except Exception as e:
        return f"AI_CHYBA: Během volání API nastala chyba: {type(e).__name__}."

def reset_session():
    st.session_state.clear()
    st.rerun()

# ==============================================================================
# --- HLAVNÍ LOGIKA APLIKACE ---
# ==============================================================================

st.set_page_config(page_title="BMC Navigator", page_icon="🚀", layout="wide")

if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'initial_prompt'
    st.session_state.business_context = ""
    st.session_state.question_plan = []
    st.session_state.current_question_index = 0
    st.session_state.bmc_data = {}
    st.session_state.analysis_result = ""
    st.session_state.suggestions_result = ""

model = initialize_model()

if st.session_state.app_stage == 'initial_prompt':
    st.title("🚀 Vítejte v BMC Navigátoru")
    st.markdown("Jsem váš AI byznys kouč. Než se pustíme do samotného Business Model Canvas, potřebuji porozumět vašemu podnikání.")
    
    st.session_state.business_context = st.text_area(
        "**Popište prosím vaši firmu, její současný byznys model a případný scénář, který chcete řešit (např. expanze, změna modelu, vstup na nový trh).**",
        height=250,
        key="business_context_input"
    )

    if st.button("Pokračovat k plánu otázek", type="primary"):
        if len(st.session_state.business_context.strip()) < 50:
            st.warning("Prosím, poskytněte podrobnější popis, aby mohly být otázky co nejrelevantnější.")
        else:
            st.session_state.app_stage = 'generating_plan'
            st.rerun()

elif st.session_state.app_stage == 'generating_plan':
    with st.spinner("Děkuji za informace. Připravuji pro vás personalizovaný plán dotazování..."):
        prompt = LLM_EXPERT_QUESTION_PLANNER_CZ_TEMPLATE.format(business_context=st.session_state.business_context)
        plan_str = ask_gemini_sdk(model, prompt, temperature=0.2)
        
        if "AI_CHYBA" in plan_str:
            st.error(f"Nepodařilo se vytvořit plán: {plan_str}")
            st.button("Zkusit znovu", on_click=reset_session)
        else:
            try:
                cleaned_json_text = plan_str.strip().lstrip("```json").rstrip("```").strip()
                st.session_state.question_plan = json.loads(cleaned_json_text)
                st.session_state.app_stage = 'questioning'
                st.rerun()
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Chyba při zpracování plánu od AI: {e}\n\nOdpověď od AI:\n{plan_str}")
                st.button("Zkusit znovu", on_click=reset_session)

elif st.session_state.app_stage == 'questioning':
    idx = st.session_state.current_question_index
    plan = st.session_state.question_plan
    
    if idx < len(plan):
        q_config = plan[idx]
        st.progress((idx + 1) / len(plan), text=f"Oblast {idx + 1} z {len(plan)}")
        st.subheader(f"{q_config.get('key', 'Neznámý blok').replace('_', ' ').title()}")
        
        st.markdown(f"### {q_config.get('question', '')}")

        with st.container(border=True):
            st.markdown("###### Pro komplexní odpověď zvažte:")
            for point in q_config.get('coverage_points', []):
                st.markdown(f"- {point}")
            st.markdown("---")
            st.markdown(f"**Příklady:** *{', '.join(q_config.get('examples', []))}*")

        answer = st.text_area("Vaše odpověď:", key=f"answer_{idx}", height=200)

        col1, col2, _ = st.columns([1, 1, 5])
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
                st.session_state.bmc_data[q_config.get('key')] = "Přeskočeno"
                st.session_state.current_question_index += 1
                st.rerun()
    else:
        st.success("Skvělá práce! Zmapovali jsme celý váš byznys model.")
        st.session_state.app_stage = 'analysis'
        st.rerun()

elif st.session_state.app_stage == 'analysis':
    with st.spinner("Provádím hloubkovou strategickou analýzu vašich odpovědí..."):
        bmc_data_string = "\n".join([f"- {key}: {value}" for key, value in st.session_state.bmc_data.items() if value != "Přeskočeno"])
        analysis_prompt = LLM_DEEP_ANALYSIS_PERSONA_V2_CZ_TEMPLATE.format(bmc_data_string=bmc_data_string)
        st.session_state.analysis_result = ask_gemini_sdk(model, analysis_prompt, temperature=0.8)

    with st.spinner("Na základě analýzy generuji inovativní návrhy..."):
        suggestion_prompt = LLM_INNOVATION_SUGGESTION_PERSONA_V2_CZ_TEMPLATE.format(
            bmc_data_string=bmc_data_string,
            analysis_summary=st.session_state.analysis_result
        )
        st.session_state.suggestions_result = ask_gemini_sdk(model, suggestion_prompt, temperature=1.2)
    
    st.session_state.app_stage = 'done'
    st.rerun()

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
