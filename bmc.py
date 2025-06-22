# bmc_navigator.py
#
# Jednosouborová verze Business Model Canvas Navigátoru.
# Pro spuštění je nutné mít ve stejném adresáři soubor `.env` s GOOGLE_API_KEY
# a nainstalované knihovny z `requirements.txt`.

import os
import json
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv

# ==============================================================================
# --- KONFIGURAČNÍ SEKCE ---
# Parametry jsou zachovány přesně podle původního Colab notebooku.
# ==============================================================================

# Prioritizovaný seznam modelů
PRIORITY_MODEL_STEMS = [
    "gemini-2.5-flash-preview-05-20",  # Specifický preview model
    "gemini-1.5-flash-latest",          # Fallback 1
    "gemini-1.5-pro-latest",            # Fallback 2
    "gemini-pro",                       # Fallback 3
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

# Globální proměnná pro model
model = None

# ==============================================================================
# --- POMOCNÉ FUNKCE (UI a API) ---
# ==============================================================================

def ai_box(text: str, title: str):
    """Zobrazí zprávu od AI ve formátovaném rámečku v terminálu."""
    print("\n" + "="*80)
    print(f"🤖 {title.upper()}")
    print("-"*80)
    wrapped_text = textwrap.fill(text, width=78)
    print(wrapped_text)
    print("="*80 + "\n")

def display_user_response(text: str):
    """Zobrazí odpověď uživatele v terminálu."""
    print("\n" + "-"*80)
    print("👤 VAŠE ODPOVĚĎ:")
    wrapped_text = textwrap.fill(text, width=78)
    print(wrapped_text)
    print("-"*80 + "\n")

def user_prompt_box(main_question: str, coverage_points: list, it_examples: list) -> str:
    """Zobrazí komplexní otázku a požádá uživatele o vstup v terminálu."""
    print("\n" + "*"*80)
    print(textwrap.fill(f"❓ {main_question}", width=78))

    if coverage_points:
        print("\n   Pro komplexní odpověď zvažte prosím následující body:")
        for point in coverage_points:
            print(f"     • {textwrap.fill(point, width=72, subsequent_indent='       ')}")

    if it_examples:
        print(f"\n   Například: {', '.join(it_examples)}.")

    print("*"*80)
    response = input(">> Váš vstup (nebo napište 'skip' pro přeskočení): ")
    return response

def display_llm_output(title: str, text: str):
    """Zobrazí finální výstup od LLM (analýzu, návrhy)."""
    ai_box(text, title)

def display_status_message(text: str):
    """Zobrazí jednoduchou stavovou zprávu."""
    print(f"... [INFO] {text}")

def ask_gemini_sdk(prompt_text: str, temperature: float = None) -> str:
    """Odešle prompt na Gemini model a vrátí textovou odpověď."""
    global model
    if not model:
        return "AI_ERROR: Model not initialized. Please ensure Cell 1 executed correctly."

    config_overrides = {}
    if temperature is not None:
        config_overrides['temperature'] = float(temperature)
        display_status_message(f"AI is thinking with custom temperature: {config_overrides['temperature']}...")
    else:
        display_status_message("AI is thinking with default temperature...")

    try:
        response = model.generate_content(prompt_text, generation_config=config_overrides)

        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason.name
            return f"AI_ERROR: Your prompt was blocked for safety reasons ({reason}). Please rephrase your input."
        else:
            return "AI_ERROR: Received an incomplete response from the model. Please try again."
    except Exception as e:
        display_status_message(f"ERROR during API call: {e}")
        return f"AI_ERROR: An unexpected error occurred: {type(e).__name__}. Please check console."

# ==============================================================================
# --- FÁZE APLIKACE ---
# ==============================================================================

def display_welcome_and_introduction():
    """Zobrazí úvodní zprávu."""
    welcome_text = (
        "Welcome to the BMC Navigator! I'm your AI business coach, here to help you map and "
        "innovate your IT business model using the powerful Business Model Canvas (BMC). "
        "We will begin by exploring your business model block by block."
    )
    ai_box(welcome_text, title="🚀 Welcome Aboard!")

def get_user_input_with_llm_validation(bmc_block_question: str, block_name: str, coverage_points: list, it_examples: list) -> str:
    """Zeptá se uživatele a validuje odpověď."""
    while True:
        user_response = user_prompt_box(bmc_block_question, coverage_points, it_examples)
        user_response_stripped = user_response.strip()

        display_user_response(user_response_stripped)

        if user_response_stripped.lower() in ["n/a", "none", "skip"]:
            ai_box(f"Understood. We'll skip '{block_name}' for now.", title="✅ Acknowledged")
            return "Skipped"

        if len(user_response_stripped) < 25:
            ai_box("Děkuji. To je dobrý začátek, ale zkusme přidat více detailů k jednotlivým bodům.", title=" digging deeper...")
        else:
            return user_response_stripped

def generate_question_plan() -> list:
    """Vygeneruje dynamický plán otázek pomocí Gemini."""
    ai_box("Jsem expert na metodologii Business Model Canvas. Připravuji pro vás personalizovaný plán dotazování, abychom prozkoumali váš byznys do hloubky.", title="🧠 Příprava Plánu")
    response_text = ask_gemini_sdk(LLM_EXPERT_QUESTION_PLANNER, temperature=0.2)

    if "AI_ERROR" in response_text:
        ai_box(f"Nepodařilo se mi vytvořit plán: {response_text}", title="❌ Chyba Plánu")
        return []

    try:
        cleaned_json_text = response_text.strip().lstrip("```json").rstrip("```").strip()
        question_plan = json.loads(cleaned_json_text)
        if isinstance(question_plan, list) and all('key' in item and 'question' in item and 'coverage_points' in item for item in question_plan):
            ai_box(f"Plán dotazování byl úspěšně vygenerován. Zeptám se vás na {len(question_plan)} klíčových oblastí.", title="✅ Plán Připraven")
            return question_plan
        else:
            raise ValueError("Vygenerovaný JSON postrádá požadované klíče.")
    except (json.JSONDecodeError, ValueError) as e:
        ai_box(f"Nastala chyba při zpracování vygenerovaného plánu: {e}.", title="❌ Chyba Zpracování")
        return []

def conduct_dynamic_bmc_analysis(question_plan: list) -> dict:
    """Provede uživatele sadou dynamicky generovaných otázek."""
    ai_box("Nyní společně projdeme jednotlivé bloky vašeho byznys modelu do hloubky.", title="🚀 Jdeme na to!")
    bmc_data = {}
    for i, config in enumerate(question_plan):
        display_status_message(f"Oblast {i+1} z {len(question_plan)}: {config.get('key', 'Neznámý blok').replace('_', ' ').title()}")
        response = get_user_input_with_llm_validation(
            bmc_block_question=config.get('question', 'Chybí text otázky.'),
            block_name=config.get('key', f'Otázka {i+1}'),
            coverage_points=config.get('coverage_points', []),
            it_examples=config.get('examples', [])
        )
        bmc_data[config.get('key', f'custom_question_{i+1}')] = response
    ai_box("Skvělá práce! Zmapovali jsme celý váš byznys model.", title="🎉 Hotovo!")
    return bmc_data

def perform_llm_bmc_analysis(bmc_data: dict) -> str:
    """Odešle sebraná data k analýze."""
    display_status_message("Initiating expert strategic analysis...")
    bmc_data_string = "\n".join([f"- {key}: {value}" for key, value in bmc_data.items() if value != "Skipped"])
    analysis_prompt = f"{LLM_DEEP_ANALYSIS_PERSONA_V2}\n\nHere is the BMC data from the user:\n{bmc_data_string}"
    return ask_gemini_sdk(analysis_prompt, temperature=0.8)

def generate_llm_suggestions(bmc_data_str: str, analysis_summary: str) -> str:
    """Požádá o návrhy na základě dat a analýzy."""
    display_status_message("Generating innovation proposals based on expert analysis...")
    suggestion_prompt = (
        f"{LLM_INNOVATION_SUGGESTION_PERSONA_V2}\n\n"
        f"**User's Business Model Canvas Data:**\n{bmc_data_str}\n\n"
        f"**Strategic Analysis Summary:**\n{analysis_summary}\n\n"
        "Now, generate the innovation suggestions based on all the above information."
    )
    return ask_gemini_sdk(suggestion_prompt, temperature=1.2)


# ==============================================================================
# --- HLAVNÍ SPUŠTĚCÍ BLOK ---
# ==============================================================================

def run_main_session():
    """Řídí celý interaktivní proces od začátku do konce."""
    global model

    # Fáze 1: Úvod a generování plánu
    display_welcome_and_introduction()
    question_plan = generate_question_plan()

    if not question_plan:
        ai_box("Nepodařilo se mi připravit plán dotazování. Zkuste prosím spustit sezení znovu.", title="❌ Chyba Spuštění")
        return

    # Fáze 2: Dynamické dotazování
    current_bmc_data = conduct_dynamic_bmc_analysis(question_plan)

    # Fáze 3: Analýza
    analysis_result = perform_llm_bmc_analysis(current_bmc_data)
    display_llm_output("Fáze 3: Strategická Analýza", analysis_result)
    input("Stiskněte Enter pro pokračování k Návrhům Inovací...")

    # Fáze 4: Návrhy
    bmc_summary_str = "\n".join([f"- {k}: {v}" for k, v in current_bmc_data.items() if v != "Skipped"])
    suggestions_result = generate_llm_suggestions(bmc_summary_str, analysis_result)
    display_llm_output("Fáze 4: Návrhy Inovací", suggestions_result)

    # Závěr
    ai_box(
        "Tímto končí naše interaktivní sezení s BMC Navigátorem. Analýza a návrhy byly založeny na expertní znalosti standardní Business Model Canvas metodologie.",
        title="🎉 Sezení Dokončeno!"
    )

if __name__ == "__main__":
    # --- Autentizace a Inicializace ---
    print("Starting API Key Configuration...")
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key not found. Please create a .env file and set the GOOGLE_API_KEY variable.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Google Generative AI API configured.")

    try:
        print("\nSearching for an available Gemini model...")
        model_name_to_use = None
        available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

        for model_stem in PRIORITY_MODEL_STEMS:
            found_model = next((m for m in available_models if model_stem in m.name and 'vision' not in m.name.lower()), None)
            if found_model:
                model_name_to_use = found_model.name
                print(f"  > Found priority model: {model_name_to_use}")
                break

        if not model_name_to_use:
            raise ValueError("Could not find any of the priority models. Please check available models and project access.")

        model = genai.GenerativeModel(
            model_name=model_name_to_use,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        print(f"Model '{model_name_to_use}' initialized successfully.")

        # --- Spuštění Hlavního Sezení ---
        run_main_session()

    except Exception as e:
        print(f"\nCRITICAL ERROR during initialization or execution: {e}")
        print("Please check your API key, model availability in your region, and project quotas.")
