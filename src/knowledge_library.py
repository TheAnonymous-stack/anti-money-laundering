"""Load and query the AML Knowledge Library"""
import json
import os

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_library", "aml_red_flags.json")

def load_red_flags():
    """Load all red flags from the JSON file"""
    with open(LIBRARY_PATH, "r") as f:
        return json.load(f)

def get_by_typology(typology):
    """Return all red flags matching a typology"""
    flags = load_red_flags()
    return [red_flag for red_flag in flags if red_flag["typology"] == typology]

def get_by_severity(severity):
    """Return all red flags at a given severity level"""
    flags = load_red_flags()
    return [red_flag for red_flag in flags if red_flag["severity"] == severity]

def get_by_customer_type(customer_type):
    """Return red flags applicable to the specified customer type"""
    flags = load_red_flags()
    return [red_flag for red_flag in flags if customer_type in red_flag["customer_type"]]