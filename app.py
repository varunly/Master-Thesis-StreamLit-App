# ==============================================================
# SAMPLE METADATA EXTRACTION (UPDATED FOR YOUR FORMAT)
# ==============================================================
def extract_thickness(filename: str) -> Optional[float]:
    """
    Extract thickness from filename
    Examples: 
    - "UL_5mm_..." -> 5.0 (upper layer)
    - "LL_10mm_..." -> 10.0 (lower layer)
    - "5mm_sample.asc" -> 5.0
    - "thickness_10mm.asc" -> 10.0
    """
    patterns = [
        r'UL[_\s-]*(\d+\.?\d*)\s*mm',  # UL_5mm
        r'LL[_\s-]*(\d+\.?\d*)\s*mm',  # LL_10mm
        r'(\d+\.?\d*)\s*mm',            # 5mm, 10mm
        r't[_\s-]*(\d+\.?\d*)',         # t5, t_10
        r'thickness[_\s-]*(\d+\.?\d*)', # thickness_5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_concentration(filename: str) -> Optional[Dict[str, float]]:
    """
    Extract concentration from filename (handles upper and lower layers)
    Examples:
    - "UL_5%IL_LL_1%IL" -> {'upper': 5.0, 'lower': 1.0}
    - "5%_sample.asc" -> {'upper': 5.0, 'lower': None}
    Returns dict with 'upper' and 'lower' keys
    """
    conc_data = {'upper': None, 'lower': None}
    
    # Pattern for UL_5%IL (upper layer)
    ul_pattern = r'UL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ul_match = re.search(ul_pattern, filename, re.IGNORECASE)
    if ul_match:
        conc_data['upper'] = float(ul_match.group(1))
    
    # Pattern for LL_1%IL (lower layer)
    ll_pattern = r'LL[_\s-]*(\d+\.?\d*)\s*%\s*IL'
    ll_match = re.search(ll_pattern, filename, re.IGNORECASE)
    if ll_match:
        conc_data['lower'] = float(ll_match.group(1))
    
    # Fallback patterns for simpler formats
    if conc_data['upper'] is None and conc_data['lower'] is None:
        simple_patterns = [
            r'(\d+\.?\d*)\s*%',          # 5%, 10.5%
            r'(\d+)p(\d+)',              # 1p5 -> 1.5
            r'c[_\s-]*(\d+\.?\d*)',      # c5, c_10
            r'conc[_\s-]*(\d+\.?\d*)',   # conc_5
        ]
        
        for pattern in simple_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                if 'p' in pattern and len(match.groups()) > 1:
                    conc_data['upper'] = float(f"{match.group(1)}.{match.group(2)}")
                else:
                    conc_data['upper'] = float(match.group(1))
                break
    
    # Return None if no concentration found
    if conc_data['upper'] is None and conc_data['lower'] is None:
        return None
    
    return conc_data

def extract_nd(filename: str) -> float:
    """
    Extract ND/OD filter value from filename
    Examples:
    - "OD=2" -> 2.0
    - "ND2" -> 2.0
    - "ND=3.5" -> 3.5
    """
    patterns = [
        r'OD\s*[=_-]*\s*(\d+\.?\d*)',   # OD=2, OD_2
        r'ND\s*[=_-]*\s*(\d+\.?\d*)',   # ND=2, ND_2
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0

def extract_qs(filename: str) -> float:
    """
    Extract Q-switch value from filename
    Examples:
    - "QS_110" -> 110.0
    - "QS150" -> 150.0
    """
    patterns = [
        r'QS[_\s-]+(\d+\.?\d*)',        # QS_110, QS 110
        r'QS(\d+\.?\d*)',                # QS110
        r'qs[_\s-]*(\d+\.?\d*)',         # qs_110
        r'q[_\s-]*(\d+\.?\d*)',          # q_110
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return np.nan

def extract_dye_amount(filename: str) -> Optional[float]:
    """
    Extract dye amount from filename
    Examples:
    - "17mgR6G" -> 17.0
    - "10mg_R6G" -> 10.0
    """
    patterns = [
        r'(\d+\.?\d*)\s*mg\s*R6G',      # 17mgR6G, 10mg R6G
        r'(\d+\.?\d*)\s*mg',             # 17mg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_repetitions(filename: str) -> Optional[int]:
    """
    Extract number of repetitions from filename
    Examples:
    - "10rep" -> 10
    - "5_rep" -> 5
    """
    patterns = [
        r'(\d+)\s*rep',                  # 10rep, 10 rep
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def get_sample_label(thickness: Optional[float], concentration: Optional[Dict], 
                    dye_amount: Optional[float] = None) -> str:
    """
    Generate a comprehensive label for the sample
    """
    parts = []
    
    if thickness is not None:
        parts.append(f"UL {thickness}mm")
    
    if concentration is not None:
        if concentration.get('upper') is not None and concentration.get('lower') is not None:
            parts.append(f"UL {concentration['upper']}%IL | LL {concentration['lower']}%IL")
        elif concentration.get('upper') is not None:
            parts.append(f"{concentration['upper']}%IL")
        elif concentration.get('lower') is not None:
            parts.append(f"LL {concentration['lower']}%IL")
    
    if dye_amount is not None:
        parts.append(f"{dye_amount}mg R6G")
    
    if parts:
        return " | ".join(parts)
    return "No Label"

def get_short_label(thickness: Optional[float], concentration: Optional[Dict]) -> str:
    """
    Generate a short label for plotting (without dye info)
    """
    parts = []
    
    if thickness is not None:
        parts.append(f"{thickness}mm")
    
    if concentration is not None:
        if concentration.get('upper') is not None and concentration.get('lower') is not None:
            parts.append(f"UL{concentration['upper']}%-LL{concentration['lower']}%")
        elif concentration.get('upper') is not None:
            parts.append(f"{concentration['upper']}%")
    
    if parts:
        return " ".join(parts)
    return "No Label"
