from backend.thin_api import app

r'''

import re
from typing import TypedDict, Optional, List, Literal
from backend.tools import retriever

from functools import lru_cache



from langgraph.graph import START, StateGraph, END

from dotenv import load_dotenv


from langchain_groq import ChatGroq
import os

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )


# 2. DEFINE STATE
class PricingState(TypedDict):
    #-----INPUTS-----
    model_name: str
    turns_on: bool
    # Granular condition tiers
    screen_condition: Optional[Literal["Good", "Minor Scratches", "Major Scratches", "Cracked", "Shattered"]]
    has_box: bool
    has_bill: bool            # New field for Bill
    is_under_warranty: bool   # New field for Warranty

    #-----Intermediates-----
    base_price: Optional[float]
    variant_options : Optional[List[dict]]

    #-----Outputs-----
    final_price: Optional[float]
    log: List[str]

# --- NEW HELPER: ROBUST PARSING ---
def extract_decimal_from_text(text: str) -> float:
    """
    Uses Regex to find the first float value (e.g., 0.45) in a string,
    ignoring all other text.
    """
    try:
        # Match a number that looks like 0.45, .45, or 1.0
        match = re.search(r"0\.\d+|1\.0|\.\d+", text)
        if match:
            return float(match.group())
        # If no decimal found, check for whole numbers like 35 (meaning 35%)
        match_int = re.search(r"\d+", text)
        if match_int:
            val = float(match_int.group())
            return val / 100 if val > 1 else val
        return 0.0
    except:
        return 0.0

# 3. HELPER FUNCTION: DYNAMIC APPRAISAL
@lru_cache(maxsize=512)
def get_dynamic_deduction(model_name: str, issue_type: str) -> float:
    """
    Asks the AI (Groq) to estimate a fair deduction % based on the specific severity.
    """
    # Strict prompt asking for numeric output
    prompt = f"""You are an expert mobile phone appraiser specializing in the Indian resale market.

    Phone Model: {model_name}
    Screen Issue: {issue_type}

    Provide the market deduction percentage (0.0 to 1.0) for this defect based on these guidelines:

    SEVERITY SCALE:
    • Minor Scratches: 0.05-0.10 (light surface marks, fully functional)
    • Major Scratches: 0.15-0.20 (deep/multiple scratches, visible in use)
    • Cracked: 0.25-0.35 (hairline/corner cracks, touch works)
    • Shattered: 0.50-0.70 (spiderweb cracks, glass missing, display damaged)

    ADJUSTMENT FACTORS:
    1. Flagship/Premium phones (iPhone Pro, Samsung Ultra, Fold/Flip): Use HIGHER end due to expensive screen replacement costs
    2. Mid-range phones: Use middle of range
    3. Budget phones (<₹15k original price): Use LOWER end unless shattered

    Return ONLY a decimal number (e.g., 0.35). DO NOT write any text or reasoning."""
    
    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        content = response.content.strip()
        
        # USE ROBUST PARSING
        deduction = extract_decimal_from_text(content)
        
        # Sanity check: If AI returns 0.0 for severe damage, trigger fallback
        if deduction == 0.0 and issue_type in ["Cracked", "Shattered"]:
            raise ValueError("AI returned 0.0 for severe damage")
            
        return deduction

    except Exception as e:
        print(f"⚠️ AI Appraiser Error: {e}")
        # Robust Fallbacks
        if "Shattered" in issue_type: return 0.60
        if "Cracked" in issue_type: return 0.35
        if "Major" in issue_type: return 0.20
        if "Minor" in issue_type: return 0.10
        return 0.0

# 4. NODE 1: RETRIEVER
def retrieve_node(state: PricingState) -> dict:
    full_query = state["model_name"]
    
    if "," in full_query:
        model_query = full_query.split(",")[0].strip()
        user_specs = full_query.split(",")[1].strip().lower() 
    else:
        model_query = full_query
        user_specs = None

    search_result = retriever.invoke(model_query)
    lines = search_result.strip().split("\n")
    
    log = state.get("log", []).copy()
    variant_options = []
    
    for line in lines:
        if "Price:" in line and "|" in line:
            try:
                parts = line.split("|")
                price = float(parts[1].split("Price:")[-1].strip())
                name = parts[0].split("Model:")[-1].strip().lower()
                variant_options.append({"full_text": name, "price": price, "raw_line": line})
            except: continue 

    extracted_price = None

    if not variant_options:
        log.append(f"⚠️ No pricing data found for '{model_query}'")
    elif user_specs:
        clean_user_spec = user_specs.replace(" ", "")
        for option in variant_options:
            if clean_user_spec in option["full_text"].replace(" ", ""):
                extracted_price = option["price"]
                log.append(f"✅ Matched Spec '{user_specs}': {option['raw_line']}")
                break
        if extracted_price is None:
            extracted_price = variant_options[0]["price"]
            log.append(f"⚠️ Spec '{user_specs}' not found. Defaulting to best match.")
    else:
        extracted_price = variant_options[0]["price"]
        if len(variant_options) > 1:
            log.append(f"ℹ️ Multiple variants found. Selected top match.")
        
    return {"base_price": extracted_price, "log": log}

# 5. NODE 2: CALCULATOR
def calculate_node(state: PricingState) -> dict:
    price = state["base_price"]
    turns_on = state["turns_on"]
    screen_condition = state["screen_condition"]
    has_box = state["has_box"]
    has_bill = state.get("has_bill", True) # Default to True if missing
    is_warranty = state.get("is_under_warranty", False)
    
    calculated_price = price if price is not None else 0.0
    log = state.get("log", []).copy()
    
    if price is None:
        log.append("❌ Cannot calculate final price: Base price not found.")
        return {"final_price": 0.0, "log": log}
    
    # Logic 1: Power
    if not turns_on:
        log.append("❌ Mobile does not turn on. Price set to ₹0.")
        return {"final_price": 0.0, "log": log}

    # Logic 2: Screen (AI DEDUCTION)
    if screen_condition != "Good":
        deduction_factor = get_dynamic_deduction(state["model_name"], screen_condition)
        
        deduction_amount = calculated_price * deduction_factor
        calculated_price -= deduction_amount
        
        pct_str = f"{int(deduction_factor * 100)}%"
        log.append(f"📉 Condition '{screen_condition}': AI applied {pct_str} deduction (-₹{deduction_amount:.2f}).")
    else:
        log.append("✅ Screen Condition: Good (No deduction)")

    # Logic 3: Box, Bill & Warranty
    if not has_bill:
        calculated_price -= 1000
        log.append("📄 Missing Bill: -₹1,000")
        
    if not has_box:
        calculated_price -= 500
        log.append("📦 Missing Box: -₹500")
        
    if is_warranty:
        bonus = calculated_price * 0.10
        calculated_price += bonus
        log.append(f"🛡️ Under Warranty: +10% Bonus (+₹{bonus:.2f})")
    
    return {
        "final_price": max(0, calculated_price),
        "log": log
    }

# 6. BUILD GRAPH
graph = StateGraph(PricingState)
graph.add_node("Retrieve Base Price", retrieve_node)
graph.add_node("Calculate Price", calculate_node)

graph.add_edge(START, "Retrieve Base Price")
graph.add_edge("Retrieve Base Price", "Calculate Price")
graph.add_edge("Calculate Price", END)

app = graph.compile()

# --- TEST ---
if __name__ == "__main__":
    
    # TEST CASE 1: Shattered (High Deduction)
    print("\n--- TEST 1: Shattered Screen ---")
    inputs_1 = {
        "model_name": "Google Pixel 9 Pro Fold", 
        "turns_on": True,
        "screen_condition": "Shattered", 
        "has_box": True,
        "has_bill": True,
        "is_under_warranty": False,
        "base_price": None,
        "final_price": None,
        "log": []
    }
    result_1 = app.invoke(inputs_1)
    print(f"Model: {result_1['model_name']}")
    print(f"Final Price: ₹{result_1['final_price']:,.2f}")
    print(f"Reasoning: {result_1['log'][-3:]}")

    # TEST CASE 2: Minor Scratches (Low Deduction)
    print("\n--- TEST 2: Minor Scratches ---")
    inputs_2 = {
        "model_name": "iPhone 13", 
        "turns_on": True,
        "screen_condition": "Minor Scratches", 
        "has_box": True,
        "has_bill": True,
        "is_under_warranty": False,
        "base_price": None,
        "final_price": None,
        "log": []
    }
    result_2 = app.invoke(inputs_2)
    print(f"Model: {result_2['model_name']}")
    print(f"Final Price: ₹{result_2['final_price']:,.2f}")
    print(f"Reasoning: {result_2['log'][-3:]}")

'''