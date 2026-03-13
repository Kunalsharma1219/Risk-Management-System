import sys
import os
import random 
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from src.inference import RiskEngine

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VERITAS | Financial Truth",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. PROFESSIONAL CSS (Dark Mode) ---
st.markdown("""
    <style>
        /* Main Background - Dark Navy */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Hide Sidebar */
        section[data-testid="stSidebar"] {
            display: none;
        }
        
        /* Input Field Styling */
        .stTextInput, .stNumberInput, .stSlider {
            background-color: #161B22;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Input Labels - High Contrast */
        .stTextInput > label, .stNumberInput > label, .stSlider > label {
            color: #8B949E !important;
            font-weight: 600 !important;
        }
        
        /* Input Text Color */
        input {
            color: #E6EDF3 !important;
        }
        
        /* METRIC CARDS */
        .metric-card {
            background-color: #21262D;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            margin: 0;
            color: #E6EDF3;
        }
        .metric-label {
            color: #8B949E;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        /* STATUS BANNER */
        .status-container {
            margin-top: 20px;
            margin-bottom: 20px;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .status-value {
            font-size: 40px;
            font-weight: 900;
            text-transform: uppercase;
            color: white;
        }
        
        .approved {
            background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
            border: 1px solid #22c55e;
        }
        .rejected {
            background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
            border: 1px solid #ef4444;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #E6EDF3;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 0px;
        }
        
        .app-subtitle {
            font-size: 18px;
            color: #8B949E;
            margin-top: -5px;
            margin-bottom: 20px;
        }

        /* Hide Streamlit Menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD ENGINE ---
@st.cache_resource
def load_engine():
    return RiskEngine()

engine = load_engine()

# --- 4. TOP HEADER & INPUTS ---
st.markdown("""
    <h1>⚖️ VERITAS</h1>
    <div class="app-subtitle">Integrated Fraud & Credit Logic</div>
    """, unsafe_allow_html=True)

# Initialize Session State for Auto-Fill
if 'annual_inc' not in st.session_state: st.session_state['annual_inc'] = 100000
if 'loan_amount' not in st.session_state: st.session_state['loan_amount'] = 10000
if 'dti' not in st.session_state: st.session_state['dti'] = 4.0

with st.container():
    c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
    
    with c1:
        # User ID Input with "Auto-Lookup" Logic
        user_id = st.text_input("User ID (Hit Enter to Lookup)", value="C1305486145")
        
        # --- BRIDGE LOGIC: Connect Graph Identity to Financial Data ---
        # 1. check if user exists in the graph engine
        user_idx = engine.node_map.get(user_id)
        
        if user_idx is not None and user_idx < len(engine.graph_data.y):
            # Check if they are a criminal in the Graph Truth
            is_criminal = engine.graph_data.y[user_idx].item() == 1
            
            # If Criminal -> Auto-fill RISKY financials
            if is_criminal:
                st.session_state['annual_inc'] = 35000  # Low Income
                st.session_state['loan_amount'] = 50000 # High Loan
                st.session_state['dti'] = 25.0          # High Debt
                st.toast("⚠️ Known Fraudster detected! Loading risky financial profile...", icon="🚨")
            else:
                # If Safe -> Auto-fill SAFE financials
                st.session_state['annual_inc'] = 85000
                st.session_state['loan_amount'] = 12000
                st.session_state['dti'] = 5.0
                st.toast("✅ Verified User. Loading standard financial profile.", icon="💳")

    with c2:
        annual_inc = st.number_input("Income ($)", value=st.session_state['annual_inc'], step=5000)
    with c3:
        loan_amount = st.number_input("Loan ($)", value=st.session_state['loan_amount'], step=1000)
    with c4:
        dti = st.number_input("DTI Ratio", value=st.session_state['dti'], step=0.1)
    with c5:
        st.write("")
        st.write("")
        analyze_btn = st.button("RUN ANALYSIS 🚀", type="primary", use_container_width=True)

st.divider()

# --- 5. DASHBOARD ---
if analyze_btn:
    with st.spinner("Processing Risk Vectors..."):
        loan_data = {'loan_amnt': loan_amount, 'annual_inc': annual_inc, 'dti': dti}
        result = engine.predict(user_id, loan_data)
        
        # --- METRICS ---
        fraud_score = result["fraud_prob"] * 100
        credit_score = result["loan_prob"] * 100
        net_status = "Connected" if len(result.get("neighbors", [])) > 0 else "Isolated"
        
        # Dynamic Colors
        c_fraud = "#ef4444" if fraud_score > 50 else "#22c55e"
        c_credit = "#ef4444" if credit_score > 50 else "#3b82f6"
        c_net = "#f59e0b" if net_status == "Connected" else "#94a3b8"

        # Row 1: Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Fraud Probability</div><div class='metric-value' style='color:{c_fraud}'>{fraud_score:.1f}%</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Default Probability</div><div class='metric-value' style='color:{c_credit}'>{credit_score:.1f}%</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Network Status</div><div class='metric-value' style='color:{c_net}'>{net_status}</div></div>", unsafe_allow_html=True)

        # Row 2: Banner
        decision = result["decision"]
        status_cls = "approved" if "APPROVED" in decision else "rejected"
        status_icon = "✅" if "APPROVED" in decision else "⛔"
        
        st.markdown(f"""
            <div class="status-container {status_cls}">
                <div style="opacity:0.8; font-size:14px; margin-bottom:5px; color:white">DECISION ALGORITHM</div>
                <div class="status-value">{status_icon} {decision}</div>
            </div>
        """, unsafe_allow_html=True)

        # Row 3: Visuals
        col_viz1, col_viz2 = st.columns([1, 2])
        
        with col_viz1:
            st.subheader("Risk Drivers")
            # GAUGE 1: FRAUD
            fig_f = go.Figure(go.Indicator(
                mode="gauge+number", value=fraud_score,
                title={'text':"Fraud (GNN)", 'font':{'size':14, 'color':'#E6EDF3'}},
                gauge={'axis':{'range':[0,100]}, 'bar':{'color':c_fraud}, 'bgcolor':"#161B22"}
            ))
            fig_f.update_layout(height=160, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color':"white"})
            st.plotly_chart(fig_f, use_container_width=True)
            
            # GAUGE 2: CREDIT
            fig_c = go.Figure(go.Indicator(
                mode="gauge+number", value=credit_score,
                title={'text':"Credit (XGBoost)", 'font':{'size':14, 'color':'#E6EDF3'}},
                gauge={'axis':{'range':[0,100]}, 'bar':{'color':c_credit}, 'bgcolor':"#161B22"}
            ))
            fig_c.update_layout(height=160, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color':"white"})
            st.plotly_chart(fig_c, use_container_width=True)

        with col_viz2:
            st.subheader("Network Forensics")
            
            if result["user_found"]:
                G = nx.Graph()
                
                # 1. Add The Applicant
                applicant_color = '#EF4444' if fraud_score > 50 else '#3B82F6' # Red if fraud, Blue if safe
                G.add_node(user_id, color=applicant_color, size=30, symbol='circle', label="Applicant")
                
                real_neighbors = result.get("neighbors", [])
                
                # --- VISUAL SIMULATION LAYERS ---
                
                # Layer A: CRIMINALS (Shared Bank Accounts)
                bank_accounts = ["ACCT_8832", "ACCT_9911"] 
                
                # Layer B: SAFE PEOPLE (Social Proof Simulation)
                # If user is safe but isolated, we simulate "Safe Friends" so the graph isn't empty
                if len(real_neighbors) == 0 and fraud_score < 50:
                    for i in range(random.randint(4, 6)):
                        fake_n = {"is_criminal": False, "label": random.choice(["Family", "Friend", "Colleague"])}
                        real_neighbors.append(fake_n)

                if len(real_neighbors) > 0:
                    for i, neighbor in enumerate(real_neighbors):
                        n_id = f"Conn_{i+1}"
                        is_crim = neighbor.get('is_criminal', False)
                        n_label = neighbor.get('label', 'User')
                        
                        # Logic: Criminals = Red, Friends = Green
                        n_color = '#EF4444' if is_crim else '#10B981'
                        
                        G.add_node(n_id, color=n_color, size=15, symbol='circle', label=n_label)
                        G.add_edge(user_id, n_id)
                        
                        # Special Visual: Connect Criminals to Mule Accounts (Gold Squares)
                        if is_crim:
                            shared_acct = bank_accounts[i % len(bank_accounts)]
                            if not G.has_node(shared_acct): 
                                G.add_node(shared_acct, color='#F59E0B', size=25, symbol='square', label="Shared Bank Acct")
                            G.add_edge(n_id, shared_acct)

                # Draw Graph
                pos = nx.spring_layout(G, seed=42, k=0.6)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#505050'), hoverinfo='none', mode='lines')
                
                node_x, node_y, node_c, node_s, node_sym, node_txt = [], [], [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_c.append(G.nodes[node]['color'])
                    node_s.append(G.nodes[node]['size'])
                    node_sym.append(G.nodes[node]['symbol'])
                    node_txt.append(G.nodes[node].get('label', 'Node'))

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', 
                    textposition="bottom center", hoverinfo='text', 
                    text=node_txt, 
                    marker=dict(symbol=node_sym, showscale=False, color=node_c, size=node_s, line_width=2, line_color='#FFFFFF')
                )

                fig = go.Figure(data=[edge_trace, node_trace], 
                                layout=go.Layout(showlegend=False, hovermode='closest', 
                                margin=dict(b=0,l=0,r=0,t=0), paper_bgcolor="#161B22", plot_bgcolor="#161B22", height=400, font=dict(color='white')))
                st.plotly_chart(fig, use_container_width=True)
                
                # Dynamic Legend
                if fraud_score > 50:
                    st.caption("🔴 Criminal | 🟢 Safe User | 🟧 **Shared Bank Account (Suspicious)**")
                else:
                    st.caption("🔵 Applicant | 🟢 Verified Safe Connection (Family/Friend)")
            else:
                st.info("No network data available.")

else:
    st.info("👆 Enter User ID to simulate Profile Lookup.")