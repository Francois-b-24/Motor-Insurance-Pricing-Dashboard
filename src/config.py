"""
Centralized configuration: color palette, segment labels, CSS styles.
"""

COLORS = {
    "primary": "#1e3a5f",
    "secondary": "#3d7bc7",
    "accent": "#e74c3c",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "palette": ["#1e3a5f", "#3d7bc7", "#5ba3e6", "#8ec3f5", "#c4dff6",
                 "#e74c3c", "#f39c12", "#2ecc71"],
}

SEGMENT_LABELS = {
    "DrivAge_bin": "Driver Age",
    "VehAge_bin": "Vehicle Age",
    "VehPower_bin": "Vehicle Power",
    "BonusMalus_bin": "Bonus-Malus",
    "Area": "Area (Density)",
    "VehGas": "Fuel Type",
}

CSS_STYLES = """
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        border-radius: 12px; padding: 20px 24px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-card h3 { margin:0; font-size:14px; font-weight:400; opacity:0.85;
                    text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-card h1 { margin:8px 0 0 0; font-size:32px; font-weight:700; }
    .kpi-card-green {
        background: linear-gradient(135deg, #1a7a4c 0%, #2ecc71 100%);
        border-radius: 12px; padding: 20px 24px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 10px;
    }
    .kpi-card-green h3 { margin:0; font-size:14px; font-weight:400; opacity:0.85;
                          text-transform:uppercase; letter-spacing:0.5px; }
    .kpi-card-green h1 { margin:8px 0 0 0; font-size:32px; font-weight:700; }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f, #3d7bc7);
        color: white; padding: 10px 20px; border-radius: 8px;
        margin: 20px 0 15px 0; font-size: 18px; font-weight: 600;
    }
    .sidebar-hint {
        background: #f0f2f6; border-radius: 8px; padding: 10px 16px;
        margin-bottom: 12px; font-size: 14px; color: #555;
        border-left: 4px solid #3d7bc7;
    }
    .footer {
        text-align:center; padding:20px; color:#888; font-size:13px;
        border-top:1px solid #eee; margin-top:30px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
