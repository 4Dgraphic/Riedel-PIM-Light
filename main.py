"""
Riedel PIM Lite - Product Information Management System
A professional PIM tool for managing, importing, and enriching technical product data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from streamlit_option_menu import option_menu
from openai import OpenAI
import json
import requests
import io
from typing import Dict, List, Optional, Tuple
import re
import time
from PIL import Image
from tavily import TavilyClient
from pypdf import PdfReader
import extra_streamlit_components as stx

def clean_string_val(val) -> str:
    """Unbox lists and clean strings"""
    if val is None: return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    s_val = str(val).strip()
    # Check for stringified lists just in case
    if s_val.startswith("['") and s_val.endswith("']"):
        return s_val[2:-2]
    return s_val

def sanitize_db_vals(data_dict: Dict) -> Dict:
    """Sanitize data before saving to Supabase: handle empty strings and int/float issues"""
    clean_data = {}
    for k, v in data_dict.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            clean_data[k] = None
        elif isinstance(v, float) and v.is_integer():
            clean_data[k] = int(v)
        elif isinstance(v, str) and v.endswith(".0"):
            try:
                # Check if it actually represents a rounded int
                f_val = float(v)
                if f_val.is_integer():
                    clean_data[k] = int(f_val)
                else:
                    clean_data[k] = v
            except:
                clean_data[k] = v
        else:
            clean_data[k] = v
    return clean_data

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

# Page configuration
st.set_page_config(
    page_title="Riedel PIM Lite",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Search Tool
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Category to Table Mapping
CATEGORY_TABLE_MAPPING = {
    "Intercom": "products_intercom",
    "Headsets": "products_headsets",
    "SmartPanels": "products_smartpanels",
    "Punqtum": "products_punqtum",
    "Distributed Video Networks": "products_distributed_video_networks",
    "Video Edge Devices": "products_video_edge_devices",
    "Live Video Production": "products_live_video_production",
    "Audio Edge Devices": "products_audio_edge_devices",
    "Audio Processing": "products_audio_processing",
    "Services": "products_services"
}

# Riedel Brand Colors
RIEDEL_RED = "#E30613"
BACKGROUND_LIGHT = "#F9FAFB"
SIDEBAR_LIGHT = "#FFFFFF"

# =====================================================
# CUSTOM CSS STYLING
# =====================================================

def load_custom_css():
    """Load custom CSS for Riedel branding"""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ============================
           GLOBAL VARIABLES & RESET
           ============================ */
        :root {{
            --primary-red: {RIEDEL_RED};
            --bg-color: #F8FAFC;
            --text-dark: #0F172A;
            --text-gray: #64748B;
            --border-color: #E2E8F0;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-dark);
        }}
        
        /* ============================
           SIDEBAR STYLING
           ============================ */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid var(--border-color);
        }}
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {{
            font-size: 1.25rem;
            color: #1E293B;
        }}
        
        /* Sidebar Radio Buttons specific styling to mimic navigation tabs */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] {{
            width: 100%;
        }}

        div[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
            background-color: transparent;
            color: #475569;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }}

        /* This is tricky in Streamlit, leveraging active state coloring via theme config is standard, 
           but we'll try to enforce red background for checked items if possible, 
           or rely on the global accent color setting. */
        
        /* ============================
           DASHBOARD & CARDS
           ============================ */
        
        /* Metric Cards */
        [data-testid="stMetric"] {{
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-gray);
        }}
        
        [data-testid="stMetricValue"] {{
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--text-dark);
            padding: 0.25rem 0;
        }}

        /* Custom Category Card CSS Classes (used in st.markdown) */
        .category-card {{
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .category-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        
        .card-icon {{
            width: 40px;
            height: 40px;
            background-color: #F1F5F9;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            color: #64748B;
            margin-bottom: 1rem;
        }}
        
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .card-stats {{
            font-size: 0.875rem;
            color: var(--text-gray);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .progress-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-weight: 600;
            color: #94A3B8;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .custom-progress-bar {{
            height: 6px;
            background-color: #E2E8F0;
            border-radius: 3px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background-color: var(--primary-red);
            border-radius: 3px;
        }}
        
        /* ============================
           COMPONENTS
           ============================ */
        
        /* Buttons */
        .stButton > button {{
            background-color: var(--primary-red);
            color: white;
            border-radius: 6px;
            font-weight: 500;
            border: none;
            padding: 0.625rem 1.25rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.2s;
        }}
        
        .stButton > button:hover {{
            background-color: #B9050F;
            transform: translateY(-1px);
        }}
        
        /* Hide Header Toolbar (3 dots) & Footer, but keep Sidebar Toggle */
        /* [data-testid="stToolbar"] {{visibility: hidden;}} */
        /* [data-testid="stDecoration"] {{visibility: hidden;}} */
        footer {{visibility: hidden;}}
        
        /* Inputs */
        input {{
            border-radius: 6px !important;
        }}
        
        /* Status Tracker Styles */
        .step-item {{
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            font-size: 0.875rem;
            color: var(--text-gray);
        }}
        
        .step-icon {{
            margin-right: 0.75rem;
            display: flex;
            align-items: center;
        }}
        
        </style>
    """, unsafe_allow_html=True)

# =====================================================
# DATABASE CONNECTION
# =====================================================

@st.cache_resource
def init_supabase() -> Client:
    """Initialize Supabase client"""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Supabase connection failed: {str(e)}")
        st.stop()

def deep_search_specs(sku: str, product_name: str) -> str:
    """
    Perform deep content extraction using Tavily Search API.
    """
    try:
        api_key = st.secrets["tavily"]["api_key"]
        # Initialize client
        tavily_client = TavilyClient(api_key=api_key)
        
        query = f"Riedel {sku} {product_name} technical specifications datasheet"
        
        # Domain Whitelist as requested
        include_domains = [
            "bhphotovideo.com",
            "thomann.de",
            "markertek.com",
            "riedel.net",
            "cpl.tech",
            "broadcaststore.com"
        ]
        
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=include_domains
        )
        
        # Format output
        results = response.get("results", [])
        if not results:
            return "No results found from specific domains."
            
        formatted_results = []
        for r in results:
            formatted_results.append(f"--- Source: {r['url']} ---\n{r['content']}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error executing Tavily Search: {str(e)}"

@st.cache_resource
def init_openai() -> OpenAI:
    """Initialize OpenAI client"""
    try:
        api_key = st.secrets["openai"]["api_key"]
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ OpenAI connection failed: {str(e)}")
        st.stop()

# =====================================================
# DATA FETCHING FUNCTIONS
# =====================================================

@st.cache_data(ttl=300)
def get_table_count(_supabase: Client, table_name: str) -> int:
    """Get row count for a specific table"""
    try:
        response = _supabase.table(table_name).select("sku", count="exact").execute()
        return response.count if response.count else 0
    except Exception as e:
        st.warning(f"Could not fetch count for {table_name}: {str(e)}")
        return 0

@st.cache_data(ttl=300)
def get_discontinued_count(_supabase: Client, table_name: str) -> int:
    """Get count of discontinued products (based on 'note' column)"""
    try:
        response = _supabase.table(table_name).select("sku", count="exact").ilike("note", "%discontinued%").execute()
        return response.count if response.count else 0
    except:
        return 0

@st.cache_data(ttl=300)
def get_all_category_stats(_supabase: Client) -> Dict:
    """Get statistics for all categories including discontinued status"""
    stats = {}
    for category, table_name in CATEGORY_TABLE_MAPPING.items():
        count = get_table_count(_supabase=_supabase, table_name=table_name)
        disc_count = get_discontinued_count(_supabase=_supabase, table_name=table_name)
        
        stats[category] = {
            "table_name": table_name,
            "sku_count": count,
            "discontinued": disc_count,
            "active": count - disc_count,
            "enrichment": calculate_enrichment_percentage(_supabase=_supabase, table_name=table_name)
        }
    return stats

def calculate_enrichment_percentage(_supabase: Client, table_name: str) -> int:
    """Calculate data enrichment percentage for a table"""
    try:
        # Fetch a sample of rows to calculate completeness
        response = _supabase.table(table_name).select("*").limit(100).execute()
        if not response.data:
            return 0
        
        df = pd.DataFrame(response.data)
        # Exclude metadata columns
        exclude_cols = ['sku', 'created_at', 'updated_at', 'note']
        data_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not data_cols:
            return 0
        
        # Calculate percentage of non-null values
        completeness = df[data_cols].notna().sum().sum() / (len(df) * len(data_cols)) * 100
        return int(completeness)
    except Exception as e:
        return 0

@st.cache_data(ttl=60)
def get_products_for_category(_supabase: Client, table_name: str) -> pd.DataFrame:
    """Fetch all products for a specific category"""
    try:
        response = _supabase.table(table_name).select("*").execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching products: {str(e)}")
        return pd.DataFrame()

def get_table_columns(_supabase: Client, table_name: str) -> List[str]:
    """Get column names for a table, handling empty tables via OpenAPI spec"""
    try:
        # Method 1: Try fetching one row (fastest)
        response = _supabase.table(table_name).select("*").limit(1).execute()
        if response.data:
            return list(response.data[0].keys())
            
        # Method 2: Fallback to OpenAPI definition for empty tables
        # Use st.secrets to get connection details
        try:
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
            
            # PostgREST OpenAPI endpoint
            api_url = f"{url}/rest/v1/"
            headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}"
            }
            
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                definitions = resp.json().get("definitions", {})
                
                # Check for table definition (usually just table_name)
                if table_name in definitions:
                    return list(definitions[table_name]["properties"].keys())
                    
                # Check with public schema prefix
                if f"public.{table_name}" in definitions:
                    return list(definitions[f"public.{table_name}"]["properties"].keys())
                    
        except Exception as e:
            st.warning(f"Schema fetch failed: {str(e)}")
            
        return []
    except Exception as e:
        st.warning(f"Could not fetch columns for {table_name}: {str(e)}")
        return []

# =====================================================
# DATA PROCESSING FUNCTIONS
# =====================================================

def clean_column_name(col_name: str) -> str:
    """
    Convert human-readable CSV headers to snake_case database column names
    Examples:
        "Width (cm)" -> "width_cm"
        "Operating Temperature (°C)" -> "operating_temp_celsius"
        "Hot-Pluggable" -> "hot_pluggable"
    """
    # Remove special characters and convert to lowercase
    col_name = col_name.lower().strip()
    
    # Handle common unit conversions
    unit_mappings = {
        "(cm)": "_cm",
        "(mm)": "_mm",
        "(kg)": "_kg",
        "(g)": "_g",
        "(w)": "_w",
        "(v)": "_v",
        "(°c)": "_celsius",
        "(hz)": "_hz",
        "(ohm)": "_ohm",
        "(db)": "_db",
        "(m)": "_m",
        "(ru)": "_ru",
    }
    
    for unit, replacement in unit_mappings.items():
        if unit in col_name:
            col_name = col_name.replace(unit, replacement)
    
    # Remove remaining parentheses and special characters
    col_name = re.sub(r'[()°]', '', col_name)
    
    # Replace spaces and hyphens with underscores
    col_name = re.sub(r'[\s\-/]+', '_', col_name)
    
    # Remove multiple consecutive underscores
    col_name = re.sub(r'_+', '_', col_name)
    
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    
    return col_name

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all column headers in a DataFrame"""
    df.columns = [clean_column_name(col) for col in df.columns]
    return df

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No columns to boolean, treating 'Yes (details)' as True.
    """
    # Known boolean columns from schema
    boolean_cols = [
        'redundant_power_supply', 
        'hot_pluggable', 
        'rotary_encoder', 
        'touchscreen'
    ]
    
    for col in df.columns:
        # Check if it's a known boolean column OR if values look like Yes/No
        is_target = col in boolean_cols
        
        if not is_target and df[col].dtype == 'object':
            # Auto-detect if it looks like a boolean column (subset of Yes/No)
            unique_vals = [str(v).lower() for v in df[col].dropna().unique()]
            if set(unique_vals).issubset({'yes', 'no', 'true', 'false', '1', '0'}):
                is_target = True

        if is_target and df[col].dtype == 'object':
            def fuzzy_bool(x):
                if pd.isna(x) or str(x).strip() == '':
                    return None
                s = str(x).strip().lower()
                if s.startswith(('y', 't', '1')): # Yes, True, 1
                    return True
                if s.startswith(('n', 'f', '0')): # No, False, 0
                    return False
                return None
            
            df[col] = df[col].apply(fuzzy_bool)
            
    return df

def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns by extracting the first valid number.
    Handles scenarios like:
    - European decimals: "48,3" -> 48.3
    - Ranges: "16-1024" -> 16
    - Units/Text: "≤225 W" -> 225
    """
    # Define suffixes/prefixes that strongly imply a numeric column
    numeric_suffixes = (
        '_cm', '_mm', '_m', 
        '_kg', '_g', 
        '_w', '_kw', 
        '_v', '_mv', 
        '_hz', '_khz', 
        '_db', '_ohm', 
        '_ru', 
        '_celsius', '_fahrenheit'
    )
    
    numeric_prefixes = ('number_of_', 'quantity_')
    
    for col in df.columns:
        # Skip forced text columns
        if col in ['sku', 'product_name', 'description', 'model_name', 'note', 'created_at', 'updated_at']:
            continue
            
        # Check if column name implies it should be numeric
        is_target = False
        if col.endswith(numeric_suffixes) or col.startswith(numeric_prefixes):
            is_target = True
        
        # If matches target naming OR if it contains just numbers/commas/dots but failed basic conversion allow it
        if is_target and df[col].dtype == 'object':
            
            def extract_val(x):
                if pd.isna(x) or str(x).strip() == '':
                    return None
                
                s = str(x).strip()
                # Handle european decimal comma before regex
                s = s.replace(',', '.')
                
                # Regex to extract the first valid float/integer pattern
                # Matches: "-123.45", "0.5", ".5", "100"
                # Ignores surrounding text like " W", "≤", " approx"
                match = re.search(r'[-+]?\d*\.?\d+', s)
                if match:
                    try:
                        val = float(match.group())
                        # Cast to int if it's a whole number to satisfy integer columns
                        if val.is_integer():
                            return int(val)
                        return val
                    except:
                        return None
                return None

            # Apply extraction
            df[col] = df[col].apply(extract_val)
                
    return df

# =====================================================
# PDF & AI EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""

def extract_specs_with_ai(openai_client: OpenAI, pdf_text: str, column_names: List[str]) -> Dict:
    """Use OpenAI to extract product specifications from PDF text"""
    
    # Create a prompt for GPT
    prompt = f"""
You are a technical data extraction specialist for Riedel Communications products.

Extract product specifications from the following technical datasheet text and return them as a JSON object.

Target fields to extract:
{', '.join(column_names)}

Rules:
1. Return ONLY valid JSON, no additional text
2. Use exact field names from the list above
3. Convert units if necessary (e.g., "10 cm" -> 10 for width_cm field)
4. For boolean fields (hot_pluggable, etc.), use true/false
5. For numeric fields, extract only the number
6. If a field is not found, omit it from the JSON
7. Be precise and extract only factual data

Technical Datasheet Text:
{pdf_text[:4000]}

Return JSON:
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a technical data extraction expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        return json.loads(result)
    
    except json.JSONDecodeError as e:
        st.error(f"AI returned invalid JSON: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"AI extraction error: {str(e)}")
        return {}

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

def render_dashboard(supabase: Client):
    """Render the main dashboard page"""
    
    st.title("Product Categories Overview")
    st.markdown("<p style='color: #64748B; margin-bottom: 2rem;'>Manage and monitor AI data extraction progress across your product lines.</p>", unsafe_allow_html=True)
    
    # Fetch all category statistics
    with st.spinner("Loading dashboard data..."):
        stats = get_all_category_stats(_supabase=supabase)
    
    # Calculate KPIs
    total_skus = sum(cat["sku_count"] for cat in stats.values())
    total_discontinued = sum(cat["discontinued"] for cat in stats.values())
    total_active = total_skus - total_discontinued
    
    avg_enrichment = int(sum(cat["enrichment"] for cat in stats.values()) / len(stats)) if stats else 0
    enriched_count = sum(cat["sku_count"] for cat in stats.values() if cat["enrichment"] > 50)
    processing_count = sum(cat["sku_count"] for cat in stats.values() if 10 < cat["enrichment"] <= 50) # Mock logic
    
    # Display KPI Metrics (5 Columns now)
    kpi_cols = st.columns(5)
    
    metrics = [
        {"label": "Total SKUs", "value": f"{total_skus:,}", "delta": "All Categories", "color": "normal"},
        {"label": "Active SKUs", "value": f"{total_active:,}", "delta": "Available", "color": "normal"},
        {"label": "Discontinued", "value": f"{total_discontinued}", "delta": "Inactive", "color": "off"},
        {"label": "Enriched", "value": f"{enriched_count:,}", "delta": f"{int(enriched_count/total_skus*100) if total_skus > 0 else 0}%", "color": "normal"},
        {"label": "Completion", "value": f"{avg_enrichment}%", "delta": "Global Average", "color": "normal"}
    ]
    
    for col, metric in zip(kpi_cols, metrics):
        with col:
            st.metric(label=metric["label"], value=metric["value"], delta=metric["delta"])
    
    st.markdown("###") # Spacer
    
    # Charts Section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("SKUs per Category")
        chart_data = pd.DataFrame([
            {"Category": cat, "SKUs": data["sku_count"]}
            for cat, data in stats.items()
        ]).sort_values("SKUs", ascending=False)
        
        fig = px.bar(chart_data, x="Category", y="SKUs", color_discrete_sequence=[RIEDEL_RED], text="SKUs")
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, height=300, 
                         margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col_chart2:
        st.subheader("Data Quality per Category")
        enrichment_data = pd.DataFrame([
            {"Category": cat, "Enrichment": data["enrichment"]}
            for cat, data in stats.items()
        ]).sort_values("Enrichment", ascending=False)
        
        fig = go.Figure(data=[go.Bar(
            x=enrichment_data["Category"], y=enrichment_data["Enrichment"],
            text=enrichment_data["Enrichment"].apply(lambda x: f"{x}%"), textposition='outside',
            marker_color=[RIEDEL_RED if x >= 70 else '#FFA500' if x >= 40 else '#9CA3AF' for x in enrichment_data["Enrichment"]]
        )])
        fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, height=300, yaxis_range=[0, 100],
                         margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("###") # Spacer
    
    # Category Cards Grid
    st.subheader("Product Categories")
    
    # Grid Layout: 4 columns
    grid_cols = st.columns(4)
    
    for idx, (category, data) in enumerate(stats.items()):
        with grid_cols[idx % 4]:
            # SVG Icon
            icon_svg = """
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
            """
            
            # HTML Card
            st.markdown(f"""
                <div class="category-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div class="card-icon">{icon_svg}</div>
                        <div style="color: #94A3B8;">⋮</div>
                    </div>
                    <div class="card-title" title="{category}">{category}</div>
                    <div class="card-stats">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                        {data["sku_count"]} SKUs
                    </div>
                    <div style="margin-top: auto;">
                        <div class="progress-label">
                            <span>Enrichment</span>
                            <span>{data["enrichment"]}%</span>
                        </div>
                        <div class="custom-progress-bar">
                            <div class="progress-fill" style="width: {data['enrichment']}%;"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add some vertical spacing for the next row
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# =====================================================
# PAGE 2: BULK DATA IMPORT
# =====================================================

def render_bulk_import(supabase: Client):
    """Render the bulk CSV import page"""
    
    st.title("📥 Bulk Data Import")
    st.markdown("Upload CSV files to populate your product database with initial data.")
    
    # Category Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "🎯 Select Target Category",
            options=list(CATEGORY_TABLE_MAPPING.keys()),
            help="Choose which product category to import data into"
        )
    
    with col2:
        table_name = CATEGORY_TABLE_MAPPING[selected_category]
        st.info(f"**Table:** `{table_name}`")
    
    st.divider()
    
    # File Upload
    uploaded_file = st.file_uploader(
        "📄 Upload CSV File",
        type=['csv'],
        help="Upload a CSV file with product data. Headers will be automatically mapped to database columns."
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Show original headers
            with st.expander("🔍 Original CSV Headers"):
                st.write(list(df.columns))
            
            # Clean headers
            df_cleaned = clean_headers(df.copy())
            df_cleaned = convert_boolean_columns(df_cleaned)
            df_cleaned = clean_numeric_values(df_cleaned)
            
            # Show cleaned headers
            with st.expander("✨ Cleaned Headers (Database Mapping)"):
                mapping = pd.DataFrame({
                    "Original": df.columns,
                    "Mapped To": df_cleaned.columns
                })
                st.dataframe(mapping, use_container_width=True)
            
            # Preview data
            st.subheader("📊 Data Preview")
            st.dataframe(df_cleaned.head(10), use_container_width=True)
            
            # Get database columns
            db_columns = get_table_columns(_supabase=supabase, table_name=table_name)
            
            # Check for column mismatches
            csv_cols = set(df_cleaned.columns)
            db_cols = set(db_columns)
            
            missing_in_db = csv_cols - db_cols
            if missing_in_db:
                st.warning(f"⚠️ These columns from CSV don't exist in database: {', '.join(missing_in_db)}")
                st.info("These columns will be ignored during import.")
            
            # Filter to only include columns that exist in DB
            df_to_import = df_cleaned[[col for col in df_cleaned.columns if col in db_cols]]
            
            st.divider()
            
            # Import Button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            
            with col_btn2:
                if st.button("🚀 Import to Supabase", type="primary", use_container_width=True):
                    with st.spinner("Importing data..."):
                        try:
                            # Convert DataFrame to list of dicts
                            records = df_to_import.to_dict('records')
                            
                            # Clean records: Handle NaNs and convert integer-floats (16.0 -> 16) for strict Postgres types
                            cleaned_records = []
                            for r in records:
                                new_r = {}
                                for k, v in r.items():
                                    if pd.isna(v):
                                        new_r[k] = None
                                    elif isinstance(v, float) and v.is_integer():
                                        new_r[k] = int(v)
                                    else:
                                        new_r[k] = v
                                cleaned_records.append(new_r)
                            
                            # Convert to records
                            records = cleaned_records
                            
                            # Deduplicate records by 'sku' to prevent "ON CONFLICT DO UPDATE command cannot affect row a second time"
                            # We use a dict comprehension to keep the LAST occurrence of each SKU in the CSV
                            # Ensure 'sku' column exists before attempting deduplication
                            if 'sku' in df_to_import.columns:
                                unique_records_map = {str(r['sku']): r for r in records}
                                records = list(unique_records_map.values())
                            else:
                                st.warning("No 'sku' column found for deduplication. All records will be imported.")

                            if not records:
                                st.warning("No valid records to import after deduplication.")
                                return

                            # 4. Upsert to Supabase
                            # We use upsert=True to update existing records or insert new ones
                            # count="exact" gives us the number of rows affected
                            try:
                                response = supabase.table(table_name).upsert(records).execute()
                            except Exception as e:
                                # Fallback: Try chunked upload if payload is too large, but usually 21000 is the duplications
                                dupes = [r['sku'] for r in records]
                                if len(dupes) != len(set(dupes)):
                                    st.error("Internal processing error: Duplicate SKUs remaining.")
                                raise e
                            
                            st.success(f"✅ Successfully imported {len(records)} records to {table_name}!")
                            st.balloons()
                            
                            # Clear cache to refresh dashboard
                            st.cache_data.clear()
                            
                        except Exception as e:
                            st.error(f"❌ Import failed: {str(e)}")
                            st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.exception(e)
    
    else:
        # Show upload instructions
        st.markdown("""
            <div class="info-box">
                <h4>📋 Import Instructions</h4>
                <ol>
                    <li>Select the target product category from the dropdown above</li>
                    <li>Upload a CSV file containing your product data</li>
                    <li>Review the automatic header mapping</li>
                    <li>Click "Import to Supabase" to upload the data</li>
                </ol>
                <p><strong>Note:</strong> The system will automatically convert column names like "Width (cm)" to "width_cm" to match database schema.</p>
            </div>
        """, unsafe_allow_html=True)

# =====================================================
# PAGE 3: SKU EDITOR & AI EXTRACTION
# =====================================================

# =====================================================
# PAGE 4: SCHEMA MANAGER
# =====================================================

def render_schema_manager(supabase: Client):
    st.title("Schema Manager")
    st.markdown("<p style='color: #64748B;'>Extend your PIM by adding new technical specification columns to product categories.</p>", unsafe_allow_html=True)
    
    # Validation Warning
    st.info("⚠️ Ensure you have run the `migration_add_column_rpc.sql` script in Supabase first to enable this feature.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Target")
        category = st.selectbox("Select Product Category", list(CATEGORY_TABLE_MAPPING.keys()))
        table_name = CATEGORY_TABLE_MAPPING[category]
        
    # Fetch columns
    current_cols = get_table_columns(supabase, table_name)
    
    with col2:
        st.subheader(f"Current Structure ({len(current_cols)} Columns)")
        with st.expander("View Field List", expanded=False):
            st.write(current_cols)
            
    st.divider()
    
    st.subheader("Add New Column")
    st.markdown("Create a new field for this category. The system will automatically format the name.")
    
    with st.form("add_col_form"):
        c1, c2 = st.columns(2)
        with c1:
            new_col_name = st.text_input("Column Name (e.g. 'Battery Life', 'IP Rating')")
        with c2:
            new_col_type = st.selectbox("Data Type", ["TEXT", "NUMERIC", "BOOLEAN", "INTEGER"], help="Use TEXT for ranges or alphanumerics.")
            
        submitted = st.form_submit_button("➕ Add Column", type="primary")
        
        if submitted:
            if not new_col_name:
                st.error("Column name required")
            else:
                # Slugify
                slug = new_col_name.lower().replace(" ", "_").replace("-", "_")
                # Remove special chars but keep underscore
                slug = re.sub(r'[^a-z0-9_]', '', slug)
                
                # Check exist
                if slug in current_cols:
                    st.error(f"Column '{slug}' already exists!")
                else:
                    try:
                        # Call RPC
                        supabase.rpc('add_product_column', {
                            'target_table_name': table_name,
                            'column_name': slug,
                            'data_type': new_col_type
                        }).execute()
                        st.success(f"✅ Successfully added column '{slug}' to {category}")
                        st.cache_data.clear() # Clear cache so new column shows up
                        time.sleep(1.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to add column: {str(e)}")
                        st.warning("Did you run the migration script in Supabase?")

def render_ai_enrichment(supabase: Client, openai_client: OpenAI):
    """Render the AI-powered data enrichment page"""
    
    st.title("AI Data Enrichment")
    st.markdown("<p style='color: #64748B; margin-bottom: 2rem;'>Upload technical datasheets or PDFs to automatically extract product specifications and enrich your PIM database using AI-driven web search.</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'ai_extracted_data' not in st.session_state:
        st.session_state.ai_extracted_data = {}
    if 'selected_sku_data' not in st.session_state:
        st.session_state.selected_sku_data = {}
    if 'extraction_status' not in st.session_state:
        st.session_state.extraction_status = "idle" # idle, processing, done_extraction, done_search
    
    # Layout: 3 columns with specific ratios from mockup
    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    
    # ===== LEFT COLUMN: Target Selection =====
    with col_left:
        st.markdown("""
            <div style="display: flex; align-items: center; color: #64748B; font-weight: 500; font-size: 0.875rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">⚙️</span> STEP 1: TARGET
            </div>
        """, unsafe_allow_html=True)
        
        # Category Card
        st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        
        st.markdown("**Select Category**")
        selected_category = st.selectbox(
            "Select Category",
            options=list(CATEGORY_TABLE_MAPPING.keys()),
            key="ai_category",
            label_visibility="collapsed"
        )
        
        table_name = CATEGORY_TABLE_MAPPING[selected_category]
        
        st.markdown("<div style='margin-top: 1rem;'><strong>Select SKU / Product</strong></div>", unsafe_allow_html=True)
        
        # Fetch products logic (same as before)
        products_df = get_products_for_category(_supabase=supabase, table_name=table_name)
        
        selected_sku = None
        if not products_df.empty:
            sku_options = []
            for _, row in products_df.iterrows():
                sku = row.get('sku', 'N/A')
                name = row.get('product_name', row.get('model_name', 'Unnamed'))
                sku_options.append(f"{sku} | {name}")
            
            selected_sku_option = st.selectbox(
                "Select SKU / Product",
                options=[""] + sku_options,
                key="ai_sku",
                label_visibility="collapsed"
            )
            
            if selected_sku_option:
                selected_sku = selected_sku_option.split(" | ")[0]
                product_data = products_df[products_df['sku'] == selected_sku].iloc[0].to_dict()
                st.session_state.selected_sku_data = product_data
            else:
                st.markdown("""
                    <div style="text-align: center; padding: 2rem; border: 1px dashed #CBD5E1; border-radius: 8px; color: #94A3B8; font-size: 0.875rem; background-color: #F8FAFC; margin-top: 0.5rem;">
                        No SKU selected yet.
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No products found.")
            
        st.markdown('</div>', unsafe_allow_html=True) # End card
        
        st.markdown("###")
        
        # AI Info Box (Red)
        st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; padding: 1.25rem; border-radius: 12px;">
                <div style="display: flex; align-items: start;">
                    <span style="font-size: 1.5rem; margin-right: 0.75rem;">✨</span>
                    <div>
                        <h4 style="margin: 0; color: {RIEDEL_RED}; font-size: 0.95rem; font-weight: 700;">AI Web Search Enabled</h4>
                        <p style="font-size: 0.8rem; color: #7F1D1D; margin: 0.5rem 0 0 0; line-height: 1.4;">
                            System will automatically cross-reference technical data with verified web sources.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # ===== CENTER COLUMN: PDF Upload & Processing =====
    with col_center:
        st.markdown("""
            <div style="display: flex; align-items: center; color: #64748B; font-weight: 500; font-size: 0.875rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">📄</span> STEP 2: UPLOAD DATA SOURCE
            </div>
        """, unsafe_allow_html=True)
        
        # Upload Card (Compact)
        st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
        
        # Custom Upload Visual
        st.markdown("""
            <div style="margin-bottom: 1rem;">
                <div style="width: 48px; height: 48px; background-color: #F1F5F9; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.5rem auto;">
                    <span style="font-size: 1.2rem; color: #64748B;">📄</span>
                </div>
                <h3 style="margin: 0; font-size: 1rem; color: #0F172A;">Drop technical PDF here</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_pdf = st.file_uploader(
            "Upload Technical PDF",
            type=['pdf'],
            key="pdf_uploader",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True) # End card
        
        st.markdown("###")
        
        # Check if we have a SKU selected
        has_sku = bool(st.session_state.get('selected_sku_data'))
        
        if st.button("🚀 PROCESS WITH AI (PDF + WEB)", type="primary", use_container_width=True, disabled=not has_sku):
            # BLUR OVERLAY
            blur_ph = st.empty()
            blur_ph.markdown("""
                <style>
                .loading-overlay {
                    position: fixed; top: 0; left: 0; width: 100%; height: 100vh;
                    background: rgba(255, 255, 255, 0.85);
                    backdrop-filter: blur(8px); webkit-backdrop-filter: blur(8px);
                    z-index: 999999;
                    display: flex; flex-direction: column; align-items: center; justify-content: center;
                }
                .overlay-content {
                    background: white; padding: 2rem; border-radius: 16px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    text-align: center;
                }
                </style>
                <div class="loading-overlay">
                    <div class="overlay-content">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                        <h2 style="margin: 0; color: #0F172A;">AI Enrichment in Progress</h2>
                        <p style="color: #64748B; margin-top: 0.5rem;">Running PDF Extraction & Tavily Web Search...</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.session_state.extraction_status = "processing"
            
            # Prepare Target Columns (Empty Only)
            # Get current DB data
            current_data = st.session_state.get('selected_sku_data', {})
            db_columns = get_table_columns(_supabase=supabase, table_name=table_name)
            exclude_logic = ['created_at', 'updated_at', 'sku', 'id', 'image_url', 'misc_images', 'select', 'completeness']
            
            # Find which columns are empty in the DB record
            target_cols = [
                c for c in db_columns 
                if c not in exclude_logic 
                and (current_data.get(c) is None or str(current_data.get(c)).strip() == "")
            ]
            
            if not target_cols:
                blur_ph.empty() # Clear blur
                st.warning("All fields are already filled! No enrichment needed.")
                st.session_state.extraction_status = "idle"
            else:
                extracted_accumulator = {}
                
                try:
                    # PHASE 1: PDF EXTRACTION
                    if uploaded_pdf:
                        # with st.spinner("📄 Reading PDF & Extracting Data..."): # Spinner hidden by blur
                        pdf_txt = extract_text_from_pdf(uploaded_pdf)
                        if pdf_txt:
                                pdf_data = extract_specs_with_ai(openai_client, pdf_txt, target_cols)
                                # Only keep filled keys
                                for k, v in pdf_data.items():
                                    if v is not None and str(v).strip() != "":
                                        extracted_accumulator[k] = clean_string_val(v)
                    
                    # PHASE 2: TAVILY WEB SEARCH (For remaining missing fields)
                    # Recalculate missing
                    still_missing = [c for c in target_cols if c not in extracted_accumulator]
                    
                    if still_missing:
                        # with st.spinner(f"🌐 Searching Web..."):
                            # Get SKU info
                            c_sku = current_data.get('sku', '')
                            c_name = current_data.get('product_name', '')
                            
                            web_context = deep_search_specs(c_sku, c_name)
                            
                            # Reuse AI extraction with Web Text
                            web_data = extract_specs_with_ai(openai_client, web_context, still_missing)
                            
                            for k, v in web_data.items():
                                if v is not None and str(v).strip() != "":
                                    extracted_accumulator[k] = clean_string_val(v)
                finally:
                    blur_ph.empty() # Unblur
                
                # DONE
                if extracted_accumulator:
                    st.session_state.ai_extracted_data = extracted_accumulator
                    st.session_state.extraction_status = "done_search"
                    st.success(f"Enrichment Complete! Found {len(extracted_accumulator)} new fields.")
                else:
                    st.warning("AI could not find new data for the empty fields.")
                    st.session_state.extraction_status = "idle"



    # ===== RIGHT COLUMN: Status Tracker =====
    with col_right:
        st.markdown("""
            <div style="display: flex; align-items: center; color: #64748B; font-weight: 500; font-size: 0.875rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">📊</span> STATUS TRACKER
            </div>
        """, unsafe_allow_html=True)
        
        # Determine status visuals
        status = st.session_state.extraction_status
        is_idle = status == "idle"
        is_processing = status == "processing"
        is_done = status in ["done_extraction", "done_search"]
        
        # Helper for checkmarks
        def step_icon(active, done):
            if done: return "✅"
            if active: return "🔄"
            return "⚪"
        
        # Card
        st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="font-weight: 600;">AI Extraction</span>
                <span style="background-color: #F1F5F9; font-size: 0.75rem; padding: 0.25rem 0.5rem; border-radius: 4px; color: #64748B;">
                    {status.replace('_', ' ').upper()}
                </span>
            </div>
            
            <div class="custom-progress-bar" style="margin-bottom: 1.5rem;">
                <div class="progress-fill" style="width: {'100%' if is_done else '50%' if is_processing else '0%'};"></div>
            </div>
            
            <div style="font-weight: 600; margin-bottom: 1rem; font-size: 0.9rem;">Web Search Status</div>
            
            <div class="step-item" style="color: {'#0F172A' if is_done else '#94A3B8'};">
                <span class="step-icon">{step_icon(is_processing, is_done)}</span>
                Searching Riedel.net...
            </div>
            <div class="step-item" style="color: {'#0F172A' if is_done else '#94A3B8'};">
                <span class="step-icon">{step_icon(is_processing, is_done)}</span>
                Verifying technical standards...
            </div>
            <div class="step-item" style="color: {'#0F172A' if is_done else '#94A3B8'};">
                <span class="step-icon">{step_icon(is_processing, is_done)}</span>
                Gathering high-res assets...
            </div>
            
            <div style="margin-top: 2rem; border-top: 1px solid #E2E8F0; padding-top: 1rem;">
                <strong style="color: #64748B; font-size: 0.75rem; text-transform: uppercase;">Output Preview</strong>
                <div style="margin-top: 1rem;">
        """, unsafe_allow_html=True)
        
        # Interactive Preview (Skeleton or Real Data)
        if st.session_state.ai_extracted_data:
            for k, v in list(st.session_state.ai_extracted_data.items())[:3]:
                st.markdown(f"""
                    <div style="background-color: #F8FAFC; border-radius: 6px; padding: 0.5rem; margin-bottom: 0.5rem; font-size: 0.8rem;">
                        <span style="color: #64748B;">{k}:</span> <strong style="color: #334155;">{v}</strong>
                    </div>
                """, unsafe_allow_html=True)
        else:
             # Skeleton loader
             st.markdown("""
                <div style="height: 12px; background-color: #F1F5F9; border-radius: 4px; margin-bottom: 0.75rem; width: 100%;"></div>
                <div style="height: 12px; background-color: #F1F5F9; border-radius: 4px; margin-bottom: 0.75rem; width: 80%;"></div>
                <div style="height: 12px; background-color: #F1F5F9; border-radius: 4px; margin-bottom: 0.75rem; width: 60%;"></div>
             """, unsafe_allow_html=True)
             
        st.markdown('</div></div></div>', unsafe_allow_html=True) # End card columns
    
    # ===== BOTTOM SECTION: Edit Form =====
    if st.session_state.selected_sku_data:
        st.divider()
        st.markdown("### ✏️ Edit Product Data")
        
        # Get all columns for the table
        db_columns = get_table_columns(_supabase=supabase, table_name=table_name)
        
        # Create form
        # Create form
        with st.form("product_edit_form"):
            updated_data = {}
            
            # Categorize columns for better UI
            categories = {
                "📌 Core Details": [],
                "📏 Dimensions & Weight": [],
                "⚡ Technical Specs": [],
                "📝 Other": []
            }
            
            for col in db_columns:
                if col in ['created_at', 'updated_at']: continue
                
                lower_col = col.lower()
                if any(x in lower_col for x in ['sku', 'product_name', 'model_name', 'description', 'category', 'note']):
                    categories["📌 Core Details"].append(col)
                elif any(x in lower_col for x in ['width', 'height', 'depth', 'weight', 'size']):
                    categories["📏 Dimensions & Weight"].append(col)
                elif any(x in lower_col for x in ['power', 'voltage', 'temp', 'consumption', 'supply', 'port', 'interface']):
                    categories["⚡ Technical Specs"].append(col)
                else:
                    categories["📝 Other"].append(col)
            
            # Render each category
            for cat_name, cols in categories.items():
                if not cols: continue
                
                with st.expander(cat_name, expanded=(cat_name == "📌 Core Details")):
                    form_cols = st.columns(3)
                    for idx, col_name in enumerate(cols):
                        
                        with form_cols[idx % 3]:
                            ai_val = st.session_state.ai_extracted_data.get(col_name)
                            db_val = st.session_state.selected_sku_data.get(col_name)
                            default_obj = ai_val if ai_val is not None else db_val
                            
                            # Handle different input types safely
                            label = col_name.replace('_', ' ').title()
                            
                            # highlight if AI filled (simple visual indicator)
                            if ai_val is not None and str(ai_val) != str(db_val or ""):
                                # Mark with Green Icon
                                label = f"✅ {label}"
                            
                            if col_name == 'sku':
                                st.text_input(label, value=str(default_obj or ""), disabled=True)
                                updated_data[col_name] = default_obj
                            
                            elif isinstance(default_obj, bool) or col_name.startswith('is_') or col_name.startswith('has_'):
                                updated_data[col_name] = st.checkbox(label, value=bool(default_obj))
                                
                            elif any(x in col_name for x in ['width', 'height', 'depth', 'weight']):
                                try:
                                    val = float(default_obj) if default_obj is not None else 0.0
                                    updated_data[col_name] = st.number_input(label, value=val, step=0.1)
                                except (ValueError, TypeError):
                                    updated_data[col_name] = st.text_input(label, value=str(default_obj or ""))
                            
                            else:
                                updated_data[col_name] = st.text_input(label, value=str(default_obj or ""))

            st.divider()
            # Submit button must be last in the form block
            submitted = st.form_submit_button("💾 Save to Database", type="primary", use_container_width=True)
            
            if submitted:
                try:
                    # Clean up and sanitize
                    clean_data = sanitize_db_vals(updated_data)
                    
                    # Update in database
                    response = supabase.table(table_name).upsert(clean_data).execute()
                    
                    st.success("✅ Product updated successfully!")
                    st.balloons()
                    # Optional: clear state or rerun
                    
                except Exception as e:
                    st.error(f"Update failed: {str(e)}")
                    
                    # Clear cache
                    st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")

# =====================================================
# PAGE 3: PRODUCT LIBRARY (NEW)
# =====================================================

@st.dialog("Product Editor", width="large")
def edit_product_dialog(sku: str, table_name: str, _supabase: Client):
    """Modal dialog to edit product details with Web Search Enrichment"""
    # Dynamic Title
    try:
        data = _supabase.table(table_name).select("*").eq("sku", sku).execute()
        if not data.data:
            st.error("Product not found.")
            return
        product = data.data[0]
        st.header(product.get('product_name', sku))
        st.caption(f"SKU: {sku} | Table: {table_name}")
    except Exception as e:
        st.error(f"Error loading product: {str(e)}")
        return

    # Initialize session state for this dialog if needed (Streamlit dialogs rerun script)
    enrich_key = f"enrich_res_{sku}"
        
    # Merge with Enrichment Result if available
    if enrich_key in st.session_state:
        for k, v in st.session_state[enrich_key].items():
            if v is not None and v != "":
                 product[k] = v # Overlay enriched data
                     
    # 1. AI Web Enrich Button
    with st.expander("✨ AI Web Enrichment", expanded=False):
        st.markdown("Search trusted sites (B&H, Thomann, etc.) to fill missing data.")
        if st.button("🔍 Find missing info on the web"):
             with st.spinner("Searching trusted sources (B&H, Thomann, Riedel)..."):
                try:
                    # Step 1: Deep Search
                    web_context = deep_search_specs(sku, product.get('product_name', ''))
                    
                    if not web_context or "Error" in web_context:
                        st.warning(f"Search yielded limited results or error: {web_context}")
                    
                    # Step 2: OpenAI Extraction
                    client = init_openai()
                    
                    # Prepare Columns list for prompt
                    # ONLY target Empty fields
                    target_columns = [
                        k for k, v in product.items() 
                        if k not in ['created_at', 'updated_at', 'image_url', 'misc_images', 'sku', 'id', 'select', 'completeness']
                        and (v is None or str(v).strip() == "" or v == [])
                    ]
                    
                    if not target_columns:
                        st.info("Good news! All relevant fields are already filled. No enrichment needed.")
                        st.stop()
                    
                    system_prompt = f"""
                    You are a Technical PIM Assistant. I will provide you with raw text scraped from technical websites. 
                    Your job is to extract values for the following MISSING database columns: {target_columns}. 
                    Extract ONLY factual technical data for these specific fields. 
                    If the web context contains conflicting data, prioritize data from 'riedel.net' or 'bhphotovideo.com'.
                    Return the result as JSON. Keys must match the requested columns exactly.
                    """
                    
                    user_prompt = f"""
                    Product SKU: {sku}
                    Product Name: {product.get('product_name', '')}
                    
                    Web Context:
                    {web_context}
                    """
                    
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    
                    parsed_ai = json.loads(completion.choices[0].message.content)
                    
                    st.session_state[enrich_key] = parsed_ai
                    st.success(f"Found {len(parsed_ai)} fields! Form updated (green checks).")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Enrichment process failed: {e}")

    with st.form("edit_sku_form"):
        updated_data = {}
        
        # Categorize logic (Clean Labels)
        categories = {
            "Core Details": [],
            "Dimensions & Weight": [],
            "Technical Specs": [],
            "Other": []
        }
        
        keys = list(product.keys())
        for k in keys:
            if k in ['created_at', 'updated_at', 'image_url', 'misc_images']: continue
            
            lower_col = k.lower()
            if any(x in lower_col for x in ['sku', 'product_name', 'model_name', 'description', 'category', 'note']):
                categories["Core Details"].append(k)
            elif any(x in lower_col for x in ['width', 'height', 'depth', 'weight', 'size']):
                categories["Dimensions & Weight"].append(k)
            elif any(x in lower_col for x in ['power', 'voltage', 'temp', 'consumption', 'supply', 'port', 'interface']):
                categories["Technical Specs"].append(k)
            else:
                categories["Other"].append(k)
        
        for cat_name, cols in categories.items():
            if not cols: continue
            with st.expander(cat_name, expanded=(cat_name == "Core Details")):
                desc_cols = [c for c in cols if 'description' in c]
                other_cols = [c for c in cols if 'description' not in c]
                
                for dc in desc_cols:
                     val = product[dc]
                     label = dc.replace('_', ' ').title()
                     if enrich_key in st.session_state and st.session_state[enrich_key].get(dc):
                         label = f"✅ {label}"
                     updated_data[dc] = st.text_area(label, value=clean_string_val(val), height=150)
                
                col_layout = st.columns(3)
                for i, k in enumerate(other_cols):
                     v = product[k]
                     with col_layout[i % 3]:
                         label = k.replace('_', ' ').title()
                         if enrich_key in st.session_state and st.session_state[enrich_key].get(k):
                             label = f"✅ {label}"
                         
                         if isinstance(v, bool):
                             updated_data[k] = st.checkbox(label, value=v)
                         elif isinstance(v, (int, float)):
                             updated_data[k] = st.number_input(label, value=float(v) if v is not None else 0.0)
                         else:
                             updated_data[k] = st.text_input(label, value=clean_string_val(v))


        st.divider()
        if st.form_submit_button("💾 Save product data", type="primary"):
            # Sanitize Data
            clean_payload = sanitize_db_vals(updated_data)
            
            try:
                _supabase.table(table_name).update(clean_payload).eq("sku", sku).execute()
                st.success("✅ Product data saved!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Database Error: {e}")

    # Image Management Section
    st.divider()
    st.subheader("Product Media Management")
    
    # Grid Layout for Lists
    col_media_main, col_media_graphs = st.columns(2)
    
    with col_media_main:
        st.markdown("#### 🖼️ Product Images")
        st.caption("Main visual and gallery (Stored in `image_url`)")
        
        # Upload
        new_imgs = st.file_uploader("Add Images", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key=f"img_up_{sku}")
        if new_imgs and st.button("⬆️ Upload Photos", key="btn_up_imgs"):
             upload_media(new_imgs, "image", product, _supabase, sku, table_name)
        
        st.divider()
        # List
        raw_imgs = product.get('image_url', "")
        # Parse CSV string
        if raw_imgs and isinstance(raw_imgs, str):
            img_list = [x.strip() for x in raw_imgs.split(',') if x.strip()]
        else:
            img_list = []
            
        render_media_list(img_list, "image", product, _supabase, sku, table_name)
        
    with col_media_graphs:
        st.markdown("#### 📊 Technical Graphs")
        st.caption("Diagrams and Charts (Stored in `misc_images`)")
        
        new_graphs = st.file_uploader("Add Graphs", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key=f"graph_up_{sku}")
        if new_graphs and st.button("⬆️ Upload Graphs", key="btn_up_graphs"):
             upload_media(new_graphs, "graph", product, _supabase, sku, table_name)
             
        st.divider()
        # List
        raw_graphs = product.get('misc_images', [])
        # Parse List or CSV from DB
        # Note: misc_images column might be Array or Text. check logic.
        if isinstance(raw_graphs, list):
             graph_list = raw_graphs
        elif isinstance(raw_graphs, str) and raw_graphs:
             graph_list = [x.strip() for x in raw_graphs.split(',') if x.strip()]
        else:
             graph_list = []
             
        render_media_list(graph_list, "graph", product, _supabase, sku, table_name)

    st.divider()
    if st.button("Close Dialog"):
        st.session_state.p_lib_edit_target = None
        st.rerun()

def upload_media(files, media_type, product, supabase, sku, table_name):
    """Helper to process upload"""
    bucket = "product-images"
    # Ensure bucket
    try:
        buckets = supabase.storage.list_buckets()
        b_names = [b.name if hasattr(b, 'name') else b['name'] for b in buckets]
        if bucket not in b_names:
             supabase.storage.create_bucket(bucket, options={"public": True})
    except: pass
    
    clean_prod_name = re.sub(r'[^a-zA-Z0-9]', '_', product.get('product_name', 'Product'))
    clean_prod_name = re.sub(r'_+', '_', clean_prod_name)
    
    uploaded_urls = []
    
    # DETERMINE STARTING NUMBER
    # Fetch current lists to find max index
    current_images = parse_db_list(product.get('image_url', ""))
    current_graphs = parse_db_list(product.get("misc_images", [])) # Helper needed or inline
    
    # Inline logic
    if isinstance(product.get('misc_images'), list): existing_graphs = product['misc_images']
    elif isinstance(product.get('misc_images'), str): existing_graphs = parse_csv(product['misc_images'])
    else: existing_graphs = []
    
    if isinstance(product.get('image_url'), str): existing_imgs = parse_csv(product.get('image_url', ""))
    else: existing_imgs = []
    
    # Check max number in relevant list
    reference_list = existing_imgs if media_type == "image" else existing_graphs
    max_num = 0
    for u in reference_list:
        if not u: continue
        # Expected format: ..._(\d+).jpg
        # Try to match number at end of filename
        try:
             # simple split by underscore, take last part, split by dot
             fname = u.split('/')[-1]
             base_no_ext = fname.rsplit('.', 1)[0]
             num_part = base_no_ext.split('_')[-1]
             if num_part.isdigit():
                 n = int(num_part)
                 if n > max_num: max_num = n
        except: pass
        
    start_seq = max_num + 1

    # Progress
    progress = st.progress(0)
    status = st.empty()
    
    for i, f in enumerate(files):
        status.text(f"Uploading {f.name}...")
        try:
            # Resizing and Naming
            img = Image.open(f)
            if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
            
            # Sequence
            seq_num = start_seq + i
            seq_str = f"{seq_num:02d}" # 01, 02 format
            
            # Format
            if media_type == "image":
                 fname = f"RID_{sku}_{clean_prod_name}_{seq_str}.jpg"
                 subfolder = "images"
            else:
                 fname = f"RID_{sku}_graph_{seq_str}.jpg"
                 subfolder = "graphs"

            # Buffers
            full_buf = io.BytesIO()
            img.save(full_buf, format="JPEG", quality=95)
            full_buf.seek(0)
            
            aspect = img.height / img.width
            new_h = int(500 * aspect)
            thumb = img.resize((500, new_h), Image.Resampling.LANCZOS)
            thumb_buf = io.BytesIO()
            thumb.save(thumb_buf, format="JPEG", quality=85)
            thumb_buf.seek(0)
            
            path_full = f"{sku}/{subfolder}/{fname}"
            path_thumb = f"{sku}/{subfolder}/thumb/{fname}"
            
            supabase.storage.from_(bucket).upload(path_full, full_buf.read(), {"upsert": "true", "content-type": "image/jpeg"})
            supabase.storage.from_(bucket).upload(path_thumb, thumb_buf.read(), {"upsert": "true", "content-type": "image/jpeg"})
            
            url = supabase.storage.from_(bucket).get_public_url(path_full)
            uploaded_urls.append(url)
            
        except Exception as e:
            st.error(f"Error {f.name}: {e}")
            
        progress.progress((i+1)/len(files))
        
    if uploaded_urls:
         if media_type == "image":
             # image_url is TEXT -> Save as Comma Separated String
             raw = product.get('image_url', "")
             current = parse_csv(raw)
             combined = current + uploaded_urls
             val = ",".join(combined)
             col = "image_url"
         else:
             # misc_images is TEXT[] -> Save as Python List
             raw = product.get('misc_images', [])
             if isinstance(raw, list): current = raw
             elif isinstance(raw, str): current = parse_csv(raw)
             else: current = []
             combined = current + uploaded_urls
             val = combined 
             col = "misc_images"
             
         try:
             supabase.table(table_name).update({col: val}).eq("sku", sku).execute()
             st.success("Saved!")
             time.sleep(1)
             st.rerun()
         except Exception as e:
             st.error(f"Save failed: {e}")

def parse_csv(raw_str):
    if not raw_str or not isinstance(raw_str, str): return []
    return [x.strip() for x in raw_str.split(',') if x.strip()]

def parse_db_list(val):
    if isinstance(val, list): return val
    if isinstance(val, str): return parse_csv(val)
    return []

def render_media_list(url_list, media_type, product, supabase, sku, table_name):
    """Render list of images"""
    if not url_list:
        st.info("Empty")
        return
        
    for i, url in enumerate(url_list):
        if not url: continue
        clean_url = url.split('?')[0]
        fname = clean_url.split('/')[-1]
        
        # Thumb path derivation for Preview
        # We stored thumbs in 'thumb' subfolder relative to file location
        # Full: .../images/file.jpg -> .../images/thumb/file.jpg
        # This string replace is safe if structure is maintained
        if media_type == "image":
             thumb = clean_url.replace("/images/", "/images/thumb/")
        else:
             thumb = clean_url.replace("/graphs/", "/graphs/thumb/")
             
        c1, c2, c3, c4 = st.columns([1, 1.5, 0.5, 0.5])
        with c1: st.image(thumb, width=80)
        with c2: st.caption(fname)
        with c3: 
             # Use Popover to hide the st.code box
             with st.popover("📋", use_container_width=True):
                 st.code(clean_url, language="text")
                 st.caption("Click right/corner to copy.")
        with c4:
             if st.button("✖", key=f"del_{media_type}_{i}_{sku}"): # White-ish X
                  updated = [u for u in url_list if u != url]
                  col = "image_url" if media_type == "image" else "misc_images"
                  
                  # 1. DELETE FROM STORAGE
                  bucket = "product-images"
                  try:
                      # Parse path from URL
                      # URL: https://.../public/product-images/PATH
                      if f"/{bucket}/" in clean_url:
                          path = clean_url.split(f"/{bucket}/")[-1]
                          # Also delete thumb
                          # Path: sku/images/file.jpg
                          # Thumb: sku/images/thumb/file.jpg
                          # We need to construct thumb path carefully
                          # Using the same replace logic as above
                          if media_type == "image":
                              thumb_path = path.replace("/images/", "/images/thumb/")
                          else:
                              thumb_path = path.replace("/graphs/", "/graphs/thumb/")
                              
                          supabase.storage.from_(bucket).remove([path, thumb_path])
                  except Exception as ex:
                      st.warning(f"Storage delete failed: {ex}")

                  # 2. UPDATE DB
                  if media_type == "image":
                      val = ",".join(updated)
                  else:
                      val = updated # List for TEXT[]

                  supabase.table(table_name).update({col: val}).eq("sku", sku).execute()
                  st.rerun()
        st.markdown("<hr style='margin:0; opacity:0.1'>", unsafe_allow_html=True)

@st.dialog("⚡ Bulk Edit Products", width="large")
def bulk_edit_dialog(skus: List[str], table_name: str, _supabase: Client):
    """Modal for batch updating fields"""
    st.info(f"Editing {len(skus)} items. Only fields you fill out will be updated.")
    
    # Analyze table schema to find dimension columns
    width_col, height_col, depth_col, weight_col = None, None, None, None
    try:
        # Fetch structural sample
        res = _supabase.table(table_name).select("*").limit(1).execute()
        if res.data:
            keys = res.data[0].keys()
            width_col = next((k for k in keys if 'width' in k), None)
            height_col = next((k for k in keys if 'height' in k), None)
            depth_col = next((k for k in keys if 'depth' in k), None)
            weight_col = next((k for k in keys if 'weight' in k), None)
    except:
        pass
    
    with st.form("bulk_edit_form"):
        st.subheader("General Info")
        col1, col2 = st.columns(2)
        with col1:
             new_note = st.text_input("Note (e.g., Discontinued)")
             new_category = st.text_input("Category")
        with col2:
             pass 
             
        # Dimension Section (Dynamic)
        if any([width_col, height_col, depth_col, weight_col]):
            st.subheader("Dimensions & Weight")
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            
            new_width = None
            new_height = None
            new_depth = None
            new_weight = None
            
            with d_col1:
                if width_col:
                    label = width_col.replace('_', ' ').title()
                    new_width = st.number_input(label, value=0.0, step=0.1, key="new_width")
            with d_col2:
                if height_col:
                    label = height_col.replace('_', ' ').title()
                    new_height = st.number_input(label, value=0.0, step=0.1, key="new_height")
            with d_col3:
                if depth_col:
                    label = depth_col.replace('_', ' ').title()
                    new_depth = st.number_input(label, value=0.0, step=0.1, key="new_depth")
            with d_col4:
                if weight_col:
                    label = weight_col.replace('_', ' ').title()
                    new_weight = st.number_input(label, value=0.0, step=0.1, key="new_weight")

        st.divider()
        if st.form_submit_button("🚀 Apply to All Selected", type="primary"):
            updates = {}
            if new_note: updates["note"] = new_note
            if new_category: updates["category"] = new_category
            
            # Add dimension updates if value > 0 (assuming 0 is 'no change' default for number input here, 
            # or we need a better way to signaling 'no change' like checkbox? 
            # Standard pattern: If user touches it. 
            # But st.number_input has default. 
            # I'll check if it's not 0.0, assuming no one bulk sets width to 0.
            if new_width and new_width > 0: updates[width_col] = new_width
            if new_height and new_height > 0: updates[height_col] = new_height
            if new_depth and new_depth > 0: updates[depth_col] = new_depth
            if new_weight and new_weight > 0: updates[weight_col] = new_weight
            
            if updates:
                try:
                    # Supabase 'in_' filter for bulk update
                    _supabase.table(table_name).update(updates).in_("sku", skus).execute()
                    st.success(f"Updated {len(skus)} products!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Bulk update failed: {str(e)}")
            else:
                st.warning("No changes specified.")

def render_product_library(supabase: Client):
    """Render the Product Library table with inline editing"""
    
    # 1. Filters
    col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 2])
    with col_filter1:
        tabs = ["All Categories"] + list(CATEGORY_TABLE_MAPPING.keys())
        selected_tab = st.selectbox("Category", tabs, index=0)
    with col_filter2:
        hide_discontinued = st.toggle("Hide Discontinued", value=False)
        st.caption("Excludes items with 'Discontinued' in note")
    with col_filter3:
        search_query = st.text_input("Search", placeholder="🔍 SKU, Name, Description...", label_visibility="collapsed").strip().lower()

    # 2. Fetch Data
    dfs = []
    columns_to_fetch = "*" if selected_tab != "All Categories" else "sku, model_name, product_name, note, updated_at, category, image_url"

    with st.spinner("Loading library..."):
        cats_to_fetch = [selected_tab] if selected_tab != "All Categories" else CATEGORY_TABLE_MAPPING.keys()
        
        for cat in cats_to_fetch:
            table = CATEGORY_TABLE_MAPPING.get(cat)
            if not table: continue
            try:
                res = supabase.table(table).select(columns_to_fetch).execute()
                if res.data:
                    temp_df = pd.DataFrame(res.data)
                    temp_df['Table'] = table
                    if 'category' not in temp_df.columns:
                        temp_df['category'] = cat
                    
                    # Calculate Completeness
                    # Exclude meta and internal columns to get a realistic content score
                    ignore_cols = {
                        'id', 'created_at', 'updated_at', 'sku', 'image_url', 'misc_images',
                        'Table', 'category', 'select', 'completeness', 'preview'
                    }
                    data_cols = [c for c in temp_df.columns if c.lower() not in ignore_cols]
                    
                    if data_cols:
                        # Count non-null and non-empty string
                        # Some fields might be "" string which counts as notnull in pandas
                        # Convert "" to NaN
                        clean_data = temp_df[data_cols].replace(r'^\s*$', pd.NA, regex=True)
                        temp_df['Completeness'] = clean_data.notnull().mean(axis=1)
                    else:
                        temp_df['Completeness'] = 0.0
                    dfs.append(temp_df)
            except Exception: pass
                
    if not dfs:
        st.info("No products found.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Filter Data
    if hide_discontinued:
        mask_disc = ~full_df['note'].astype(str).str.contains('discontinued', case=False, na=False)
        full_df = full_df[mask_disc]
        
    if search_query:
        mask = full_df.astype(str).apply(lambda x: x.str.lower().str.contains(search_query, na=False)).any(axis=1)
        display_df = full_df[mask].reset_index(drop=True)
    else:
        display_df = full_df
        
    # Prepare Display DF
    # Add Select Column
    display_df.insert(0, "Select", False)
    
    # Process Image URL for ImageColumn (first image only)
    def get_first_thumb(val):
        if not val or not isinstance(val, str): return None
        parts = val.split(',')
        if parts:
             return parts[0].replace("/images/", "/images/thumb/").strip()
        return None
        
    display_df['Preview'] = display_df['image_url'].apply(get_first_thumb)
    
    # Ensure Completeness is clean float
    display_df['Completeness'] = display_df['Completeness'].fillna(0.0).astype(float)

    # 4. ACTION BAR (Top)
    actions_container = st.container()

    # 5. DATA EDITOR
    # Dynamic Height
    row_height = 36
    # Add 30% buffer approximately + header
    calc_height = int((len(display_df) * row_height) * 1.3) + 50
    if calc_height < 300: calc_height = 300
    
    # Dynamic Column Order
    # User Request: Product Name frozen/first.
    primary_cols = ["Select", "product_name", "Completeness", "Preview", "sku", "category", "updated_at"]
    hidden_cols = {
        "Table", "image_url", "misc_images", "created_at", "id", 
        "Completeness", "Preview", "Select" # Already in primary
    }
    
    # Identify extra columns present in the dynamic dataframe
    extra_cols = [c for c in display_df.columns if c not in primary_cols and c not in hidden_cols]
    final_col_order = primary_cols + extra_cols
    
    col_config = {
        "Select": st.column_config.CheckboxColumn("✅", width="small", default=False),
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "product_name": st.column_config.TextColumn("Product Name", width="medium"), # Give it some width
        "category": st.column_config.TextColumn("Category", disabled=True),
        "updated_at": st.column_config.DatetimeColumn("Updated", format="D MMM YY", disabled=True),
        # Progress with hidden text (format=" ")
        "Completeness": st.column_config.ProgressColumn("Status", format=" ", min_value=0, max_value=1),
        "Preview": st.column_config.ImageColumn("Img", width="small"),
        # Hide internal cols
        "Table": None,
        "image_url": None,
        "misc_images": None,
        "created_at": None,
        "id": None
    }
    
    # Define Disabled Columns (Only Metadata)
    # Allow editing description, name, prices, etc.
    disabled_cols = ["sku", "Table", "created_at", "updated_at", "Completeness", "Preview", "category", "id"]
    
    edited_df = st.data_editor(
        display_df,
        column_config=col_config,
        column_order=final_col_order,
        key="prod_editor",
        hide_index=True,
        disabled=disabled_cols,
        height=calc_height,
        use_container_width=True
    )
    
    # 6. ACTION HANDLING (Check Edits & Selection)
    
    # Check for Edits
    # Streamlit data_editor "edited_rows" is a dict {index: {col: val}}
    changes = st.session_state.get("prod_editor", {}).get("edited_rows", {})
    # Filter out pure 'Select' toggles from being counted as unsaved "Data Edits"
    data_changes = {
        idx: {k:v for k,v in row.items() if k != "Select"} 
        for idx, row in changes.items()
    }
    # Remove empty rows
    data_changes = {i: c for i, c in data_changes.items() if c}
    
    selected_rows = edited_df[edited_df["Select"] == True]
    
    with actions_container:
        # Save Button Logic
        if data_changes:
             # Use 3 columns, middle widely to catch eye? Or just prominent.
             ca, cb, _ = st.columns([1, 2, 2])
             with ca:
                 if st.button(f"💾 Update {len(data_changes)} Changes", type="primary", use_container_width=True):
                     count = 0 
                     for row_idx, row_changes in data_changes.items():
                         try:
                             tgt = display_df.iloc[row_idx]
                             clean_row_changes = sanitize_db_vals(row_changes)
                             supabase.table(tgt['Table']).update(clean_row_changes).eq("sku", tgt['sku']).execute()
                             count += 1
                         except Exception as e:
                             st.error(f"Error saving row {row_idx}: {e}")
                     st.success(f"Saved {count} items.")
                     time.sleep(1)
                     st.rerun()
             with cb:
                 st.caption("You have unsaved changes in the table.")
                 
        # Selection Logic
        elif not selected_rows.empty:
            c1, c2, c3 = st.columns([1.5, 1.5, 4])
            with c1:
                # Primary Edit Action
                if len(selected_rows) == 1:
                     if st.button("✏️ Edit 1 Item", type="primary", use_container_width=True):
                         row = selected_rows.iloc[0]
                         st.session_state.p_lib_edit_target = {"sku": row['sku'], "table": row['Table']}
                         st.rerun()
                else:
                     if st.button(f"✏️ Edit {len(selected_rows)} Items", type="primary", use_container_width=True):
                         tables = selected_rows['Table'].unique()
                         if len(tables) > 1: st.warning("One category only.")
                         else: bulk_edit_dialog(selected_rows['sku'].tolist(), tables[0], supabase)
            with c2:
                 st.caption(f"{len(selected_rows)} selected")
        else:
             st.write("") # Spacer
    
    # Trigger Dialog
    if st.session_state.get("p_lib_edit_target"):
        target = st.session_state.p_lib_edit_target
        edit_product_dialog(target['sku'], target['table'], supabase)


# =====================================================
# MAIN APPLICATION
# =====================================================

# =====================================================
# AUTHENTICATION
# =====================================================

def render_login_page(supabase: Client, cookie_manager):
    """Render a clean login screen with Cookie Persistence"""
    
    # Centering Layout
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c2:
        # Official Riedel Logo
        svg_logo = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 470.74 122.39" style="width: 100%; max-width: 250px; height: auto;"><path fill="#1a1a1a" d="M35.13,16.27L43.27,0h35.13s24.24.38,22.2,22.1c-1.11,11.89-8.37,15.84-13.15,18.54-3.46,1.96-9.73,3.62-9.73,3.62l23.45,40.07h-27.02l-23.85-40.91s27.64,1.87,27.64-17.94c0-8.9-9.93-9.22-12.87-9.22l-29.95-.05v.05h0ZM0,84.33l21.62-40.89,24.99-.05-20.68,40.94H0ZM44.46,39.5c7.08-2.95,9.41-8.12,8.59-12.6-1.54-8.35-11.5-8.74-17.58-6.55-2.81,1.01-11.8,6.84-8.57,15.19,1.97,5.08,10.49,6.89,17.56,3.95M185.18,45.33h2.18c5.91,0,9.81-.64,11.71-1.94,1.9-1.29,2.84-3.55,2.84-6.75,0-3.37-1.02-5.76-3.05-7.18-2.04-1.42-5.87-2.13-11.5-2.13h-2.18v17.99h0ZM219.9,84.33h-16.79l-17.94-31.79v31.79h-14.6V16.27h20.85c8.29,0,14.5,1.6,18.63,4.81s6.19,8.04,6.19,14.49c0,4.68-1.42,8.68-4.25,12s-6.49,5.26-10.97,5.81l18.87,30.95h.01ZM228.53,16.27h14.59v68.06h-14.59V16.27ZM258.64,84.33V16.27h40.13v12.33h-25.54v13.96h25.54v12.33h-25.54v12.33h-25.54v17.11h25.54v12.33h-40.13ZM327.58,72h2.04c7.61,0,13.16-1.73,16.64-5.21,3.48-3.47,5.22-8.97,5.22-16.5s-1.74-12.99-5.22-16.47c-3.48-3.49-9.03-5.23-16.64-5.23h-2.04v43.4h0ZM312.99,84.33V16.27h18.59c10.61,0,19.15,4.26,22.34,6.64,4.12,3.07,7.24,6.93,9.36,11.6,2.12,4.66,3.18,9.97,3.18,15.92s-1.08,11.4-3.25,16.06c-2.17,4.67-5.34,8.52-9.52,11.55-3.1,2.24-9.47,6.28-20.95,6.28h-19.75,0ZM378.64,84.33V16.27h40.14v12.33h-25.54v13.96h25.54v12.33h-25.54v17.11h25.54v12.33h-40.14ZM432.99,84.33V16.27h14.59v55.73h23.16v12.33h-37.75ZM117.27,16.27h6.4v68.06h-6.4V16.27h0ZM132.85,0h6.4v122.39h-6.4V0ZM148.09,0h6.4v84.33h-6.4V0Z"/></svg>"""

        st.markdown(f"""
            <div style="text-align: center; margin-top: 6rem; margin-bottom: 3rem;">
                <div style="margin: 0 auto 1.5rem auto;">
                    {svg_logo}
                </div>
                <h3 style="margin: 0; color: #64748B; font-weight: 500; font-size: 1.1rem; letter-spacing: 0.5px;">PIM Lite</h3>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="user@riedel.net")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            remember = st.checkbox("Remember me", value=True)
            
            st.markdown("###")
            
            if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    if res.user:
                        st.session_state['user'] = res.user
                        if remember and res.session:
                            cookie_manager.set('pim_token', res.session.access_token, key="set_auth_cookie", expires_at=None)
                        st.success("Welcome back!")
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")

def main():
    """Main application entry point"""
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize clients
    supabase = init_supabase()
    openai_client = init_openai()
    
    # Cookie Manager (Must be init at top)
    cookie_manager = stx.CookieManager(key="auth_cookies")
    auth_token = cookie_manager.get('pim_token')
    
    # Auto Login from Cookie
    if 'user' not in st.session_state and auth_token:
        try:
            res = supabase.auth.get_user(auth_token)
            if res.user:
                st.session_state['user'] = res.user
        except:
            cookie_manager.delete('pim_token')
    
    # AUTH CHECK
    if 'user' not in st.session_state:
        render_login_page(supabase, cookie_manager)
        return
    
    # Sidebar Navigation
    with st.sidebar:
        # Sidebar Logo
        st.image("https://hdcnohbbsqjvfuccrmso.supabase.co/storage/v1/object/public/assets/Logo%20Riedel%20RGB.png", width=130)

        
        # Modern Button-based Navigation
        if 'page_nav' not in st.session_state:
            st.session_state['page_nav'] = "Dashboard"
            
        def nav_button(label, page_key):
            is_active = st.session_state['page_nav'] == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
                st.session_state['page_nav'] = page_key
                st.rerun()
                
        nav_button("Dashboard", "Dashboard")
        # Add some spacing CSS to make buttons look like menu items (less gap)
        st.markdown("""
            <style>
            /* Reduce gap between buttons */
            div.row-widget.stButton {
                margin-bottom: -10px;
            }
            
            /* INACTIVE Buttons (Secondary) -> Red Outline */
            section[data-testid="stSidebar"] button[kind="secondary"] {
                background-color: transparent !important;
                border: 1px solid #E30613 !important;
                color: #E30613 !important;
                width: 100%;
            }
            
            /* ACTIVE Button (Primary) -> Red Fill */
            section[data-testid="stSidebar"] button[kind="primary"] {
                background-color: #E30613 !important;
                border: 1px solid #E30613 !important;
                color: white !important;
                width: 100%;
            }
            
            /* Hover Interaction */
            section[data-testid="stSidebar"] button[kind="secondary"]:hover {
                background-color: #FFF0F0 !important;
                border-color: #E30613 !important;
                color: #E30613 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        nav_button("Product Library", "Product Library")
        nav_button("Bulk Import", "Bulk Import")
        nav_button("AI Enrichment", "AI Enrichment")
        nav_button("Schema Manager", "Schema Manager")
        
        page = st.session_state['page_nav']
        
        st.divider()
        
        # User Profile & Logout
        if 'user' in st.session_state:
            user_email = st.session_state['user'].email
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem; color: #475569; font-size: 0.85rem;">
                    <div style="width: 24px; height: 24px; background: #E2E8F0; border-radius: 50%; display: flex; alignItems: center; justifyContent: center; margin-right: 0.5rem;">👤</div>
                    <div style="text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{user_email}</div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Logout", use_container_width=True):
                supabase.auth.sign_out()
                if 'user' in st.session_state:
                    del st.session_state['user']
                cookie_manager.delete('pim_token')
                st.rerun()
        
        st.divider()
        
        # Footer
        st.markdown("""
            <div style="
                position: fixed;
                bottom: 1rem;
                padding: 1rem;
                border-radius: 8px;
                font-size: 0.75rem;
                color: #6B7280;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="width: 8px; height: 8px; background-color: #10B981; border-radius: 50%; margin-right: 0.5rem;"></div>
                    <span>AI Engine Online</span>
                </div>
                <div>PIM Lite v1.1.0</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Route to selected page
    if page == "Dashboard":
        render_dashboard(supabase)
    elif page == "Product Library":
        render_product_library(supabase)
    elif page == "Bulk Import":
        render_bulk_import(supabase)
    elif page == "AI Enrichment":
        render_ai_enrichment(supabase, openai_client)
    elif page == "Schema Manager":
        render_schema_manager(supabase)

# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    main()
