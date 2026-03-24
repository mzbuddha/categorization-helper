import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import os
import io
import re
import json
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import copy


# ==========================================
# 0. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Data Categorization Agent", layout="wide")
CAT_ALGO_VERSION = 3 # Increment this to trigger automatic rule reset

PRESETS_FILE = "cat_presets.json"
AUDIT_LOG_FILE = "cat_audit_log.json"

def save_presets_to_file(presets):
    try:
        # Convert keys to str for JSON
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"프리셋 저장 실패: {e}")

def load_presets_from_file():
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # JSON keys are always strings. Convert value_labels keys back to int.
                for p_name in data:
                    preset = data[p_name]
                    for v_name in preset:
                        rule = preset[v_name]
                        if 'value_labels' in rule:
                            new_v = {}
                            for k, val in rule['value_labels'].items():
                                try:
                                    new_v[int(k)] = val
                                except ValueError:
                                    new_v[k] = val
                            rule['value_labels'] = new_v
                return data
        except Exception as e:
            st.error(f"프리셋 로드 실패: {e}")
    return {}

def save_audit_log_to_file(log):
    try:
        with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"감사 로그 저장 실패: {e}")

def load_audit_log_from_file():
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"감사 로그 로드 실패: {e}")
    return []

def log_action(action, details=""):
    if 'cat_audit_log' not in st.session_state:
        st.session_state.cat_audit_log = load_audit_log_from_file()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"일시": now_str, "작업": action, "상세": details}
    st.session_state.cat_audit_log.insert(0, entry)
    if len(st.session_state.cat_audit_log) > 100: # Limit log size
        st.session_state.cat_audit_log = st.session_state.cat_audit_log[:100]
    save_audit_log_to_file(st.session_state.cat_audit_log)

def parse_spss_syntax(syntax_text):
    """Parses RECODE and VALUE LABELS from SPSS syntax to reconstruct rules."""
    rules = {}
    # Recode blocks: RECODE var (...) INTO newvar.
    recode_blocks = re.finditer(r"RECODE\s+(\w+)\s+(.+?)\s+INTO\s+(\w+)\.", syntax_text, re.S | re.I)
    
    for rb in recode_blocks:
        orig = rb.group(1)
        cnt = rb.group(2)
        newv = rb.group(3)
        
        bins = []
        missing_policy = 'keep'
        missing_code = 9
        
        chunks = re.finditer(r"\(([^)]+)\)", cnt)
        for chunk_match in chunks:
            chunk = chunk_match.group(1).strip()
            if "SYSMIS" in chunk.upper():
                m = re.search(r"=\s*(\d+)", chunk)
                if m:
                    missing_policy = 'recode'
                    missing_code = int(m.group(1))
            elif "ELSE" in chunk.upper():
                m = re.search(r"=\s*(\d+)", chunk)
                if m:
                    bin_cfg = {
                        'valL': None, 'opL': '선택안함',
                        'valR': None, 'opR': '선택안함'
                    }
                    bins.append(bin_cfg)
            else:
                m = re.search(r"(.+?)\s+THRU\s+(.+?)\s*=\s*(\d+)", chunk, re.I)
                if m:
                    lo, hi, code = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
                    
                    lo_val = None
                    opL = '선택안함'
                    if lo.upper() != "LOWEST":
                        lo_val = float(lo)
                        opL = '이상'
                        
                    hi_val = None
                    opR = '선택안함'
                    if hi.upper() != "HIGHEST":
                        hi_val = float(hi)
                        opR = '이하'
                        
                    bin_cfg = {
                        'valL': lo_val, 'opL': opL,
                        'valR': hi_val, 'opR': opR
                    }
                    bins.append(bin_cfg)
            
        rules[orig] = {
            'new_var': newv,
            'label': orig,
            'bins': bins,
            'value_labels': {},
            'missing_policy': missing_policy,
            'missing_code': missing_code,
            'continuous': True
        }

    # Labels
    vlabels = re.finditer(r"VARIABLE LABELS\s+(\w+)\s+'(.*?)'", syntax_text, re.I)
    for vl in vlabels:
        tv, lab = vl.group(1), vl.group(2).replace("_범주", "").replace("''", "'")
        for r in rules:
            if rules[r]['new_var'] == tv: rules[r]['label'] = lab

    # Val Labels
    vallabs = re.finditer(r"VALUE LABELS\s+/([^\s]+)\s+(.*?)\n\s*\.\s*(?:\n|(?=EXECUTE))", syntax_text, re.S | re.I)
    for f in vallabs:
        tv = f.group(1).strip()
        cnt = f.group(2)
        lmap = {}
        for im in re.finditer(r"(\d+)\s+'(.*?)'", cnt):
            lmap[int(im.group(1))] = im.group(2).replace("''", "'")
        for r in rules:
            if rules[r]['new_var'] == tv: rules[r]['value_labels'] = lmap
            
    return rules

def parse_column_guide_labels(xl_file):
    try:
        if '컬럼가이드' not in xl_file.sheet_names:
            return {}
        df_guide = pd.read_excel(xl_file, sheet_name='컬럼가이드', header=None)
    except Exception:
        return {}
        
    start_row = -1
    start_col = -1
    for r in range(df_guide.shape[0]):
        for c in range(df_guide.shape[1]):
            val = str(df_guide.iloc[r, c]).strip()
            if 'VALUE LABELS' in val.upper():
                start_row = r
                start_col = c
                break
        if start_row != -1:
            break
            
    if start_row == -1:
        return {}
        
    labels_dict = {}
    current_vars = []
    
    for r in range(start_row + 1, df_guide.shape[0]):
        val = str(df_guide.iloc[r, start_col]).strip().replace("''", "'")
        if not val or val.lower() == 'nan':
            continue
            
        if val.startswith('/'):
            vars_part = val[1:].strip().split()
            current_vars = vars_part
            for v in current_vars:
                if v not in labels_dict:
                    labels_dict[v] = {}
        elif current_vars:
            m = re.match(r"^(\d+)\s+['\"](.*?)['\"]", val)
            if m:
                code = int(m.group(1))
                label_str = m.group(2).strip()
                for v in current_vars:
                    labels_dict[v][code] = label_str

    return labels_dict

# ==========================================
# 1. Helper Functions (Logic)
# ==========================================

def load_data_file(uploaded_file):
    """Loads CSV, Excel, or SPSS .sav files."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext == '.sav':
            df, meta = pyreadstat.read_sav(uploaded_file)
            return df, meta
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file), None
        elif ext == '.csv':
            return pd.read_csv(uploaded_file), None
        else:
            st.error(f"Unsupported file format: {ext}")
            return None, None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

def detect_continuous(series):
    """Detects if a series is continuous based on fractional values."""
    valid_data = series.dropna()
    if valid_data.empty:
        return False
    # Check if any value has a fractional part
    return any(valid_data % 1 != 0)

def calculate_epsilon(series):
    """Calculates epsilon based on data precision: eps = 10^-(d+1)"""
    valid_data = series.dropna()
    if valid_data.empty:
        return 1e-6
    
    try:
        def get_decimals(x):
            s = str(x)
            if '.' in s:
                return len(s.split('.')[1])
            return 0
        
        # Check first 500 samples for max decimals
        d = valid_data.head(500).apply(get_decimals).max()
        return 10**(-(d+1))
    except:
        return 1e-6

def is_time_variable(var_type_val):
    if not isinstance(var_type_val, str):
        return False
    return "시간" in var_type_val

def is_person_variable(var_type_val):
    if not isinstance(var_type_val, str):
        return False
    return "인원" in var_type_val


def estimate_step(series, min_step=None):
    x = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(x) < 3:
        return None

    x = np.sort(x.unique())
    diffs = np.diff(x)
    diffs = diffs[diffs > 1e-9]

    if len(diffs) == 0:
        return None

    step = float(np.median(diffs))

    # common grids
    common = [1.0, 0.5, 0.25, 0.2, 0.1, 1/60, 5/60, 10/60, 15/60]
    for c in common:
        if abs(step - c) < c * 0.05:
            step = c
            break

    if min_step is not None:
        step = max(step, min_step)

    return step


def snap_edges(edges, step):
    if step is None:
        return sorted(set(edges))

    snapped = [round(e / step) * step for e in edges]
    return sorted(set(snapped))


def suggest_bins(series, continuous, k=3, label=None, var_type_val=None, predefined_labels=None):
    valid_data = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if valid_data.empty:
        return None

    unique_vals = sorted(valid_data.unique())
    n = len(valid_data)

    # New rule: If 6 <= n <= 50, default k should not exceed 4
    if 6 <= n <= 50:
        k = min(k, 4)

    is_likert = False
    likert_max = 0
    if isinstance(var_type_val, str) and "리커트" in var_type_val:
        is_likert = True
        # Extract scale point (e.g. "4점 리커트" -> 4)
        m = re.search(r'(\d+)점', var_type_val)
        if m:
            likert_max = int(m.group(1))

    # -----------------------------------
    # 0. 저빈도(Low Cardinality) 분할 - 소수점 생성을 방지하기 위한 특별 예외처리
    # -----------------------------------
    if len(unique_vals) <= 5 or is_likert:
        if is_likert and likert_max > 0:
            target_k = likert_max
            groups = [[float(v)] for v in range(1, likert_max + 1)]
        elif is_likert:
            target_k = len(unique_vals)
            counts = valid_data.value_counts(normalize=True).sort_index()
            groups = [[val] for val in counts.index]
        else:
            target_k = min(len(unique_vals), 3) # 고유값이 3개 이하면 그 수만큼, 4~5개면 3개로 범주화
            counts = valid_data.value_counts(normalize=True).sort_index()
            groups = [[val] for val in counts.index]
        
        while len(groups) > target_k:
            group_freqs = [sum(counts[v] for v in g) for g in groups]
            
            # Find the smallest group to merge
            min_freq = min(group_freqs)
            min_idx = group_freqs.index(min_freq)
            
            if min_idx == 0:
                groups[0].extend(groups[1])
                groups.pop(1)
            elif min_idx == len(groups) - 1:
                groups[-2].extend(groups[-1])
                groups.pop(-1)
            else:
                left_freq = group_freqs[min_idx - 1]
                right_freq = group_freqs[min_idx + 1]
                # Merge with the smaller adjacent group to balance sizes
                if left_freq < right_freq:
                    groups[min_idx - 1].extend(groups[min_idx])
                    groups.pop(min_idx)
                else:
                    groups[min_idx].extend(groups[min_idx + 1])
                    groups.pop(min_idx + 1)
                    
        bins = []
        v_labels = {}
        for i, g in enumerate(groups):
            rule = {'valL': None, 'opL': '선택안함', 'valR': None, 'opR': '선택안함'}
            min_v = float(min(g))
            max_v = float(max(g))
            
            if i == len(groups) - 1 and len(g) > 1:
                # Last group with multiple values: "X 이상"
                rule['valL'] = min_v
                rule['opL'] = '이상'
            else:
                # Single values or non-last groups: "min_v 이상 max_v 이하"
                rule['valL'] = min_v
                rule['opL'] = '이상'
                rule['valR'] = max_v
                rule['opR'] = '이하'
                    
            parts = []
            if min_v == max_v:
                cv = int(min_v)
                if predefined_labels is not None and cv in predefined_labels:
                    v_labels[i+1] = predefined_labels[cv]
                elif is_likert:
                    v_labels[i+1] = f"{cv}점"
                else:
                    v_labels[i+1] = f"{cv}"
            else:
                if rule['opL'] != '선택안함': parts.append(f"{rule['valL']} {rule['opL']}")
                if rule['opR'] != '선택안함': parts.append(f"{rule['valR']} {rule['opR']}")
                v_labels[i+1] = " ".join(parts) if parts else "전체"
            bins.append(rule)
            
        return {
            'continuous': continuous,
            'k': len(bins),
            'bins': bins,
            'value_labels': v_labels,
            'spss_notes': {
                'epsilon_used': continuous,
                'epsilon_value': calculate_epsilon(series) if continuous else 1.0
            }
        }
    else:
        # -----------------------------------
        # 1. Step 결정
        # -----------------------------------
        step = None
        
        if is_person_variable(var_type_val):
            est = estimate_step(valid_data, min_step=1.0)
            step = est if est is not None else 1.0
        elif continuous:
            if is_time_variable(var_type_val):
                step = estimate_step(valid_data, min_step=0.5)
            else:
                step = estimate_step(valid_data)
        else:
            est = estimate_step(valid_data, min_step=1.0)
            step = est if est is not None else 1.0
    
        # -----------------------------------
        # 2. 엣지(구간 경계) 초기화 및 단일값 감지
        # -----------------------------------
        valid_min = float(valid_data.min())
        valid_max = float(valid_data.max())
        
        # [핵심 수정] 최솟값이 컷포인트를 삼키는 것을 막기 위해 하한선을 step만큼 내림 (가짜 지하 1층)
        lower_bound = valid_min - step if step else valid_min - 1e-6
        edges = [lower_bound, valid_max]

        value_counts = valid_data.value_counts(normalize=True)
        top_value = value_counts.idxmax()
        top_ratio = value_counts.max()
    
        point_mass = top_value if top_ratio >= 0.5 else None

        if point_mass is not None:
            # 50% 단일값을 완벽히 분리하기 위해 좌우 엣지를 강제 추가
            pm_left = point_mass - step if step else point_mass - 1e-6
            edges.extend([float(pm_left), float(point_mass)])
            
            remaining = valid_data[valid_data != point_mass]
            rem_k = max(1, k - 1)
        else:
            remaining = valid_data
            rem_k = max(1, k)

        # -----------------------------------
        # 3. 누적 비율 탐색 분할 (Cumulative Search)
        # -----------------------------------
        if len(remaining) > 0 and rem_k > 1:
            counts = remaining.value_counts().sort_index()
            cum_pct = counts.cumsum() / counts.sum()
            
            for i in range(1, rem_k):
                target = i / rem_k
                closest_val = (cum_pct - target).abs().idxmin()
                edges.append(float(closest_val))

        # -----------------------------------
        # 4. Step 스냅 및 정렬, 중복 제거
        # -----------------------------------
        if step:
            edges = [round(e / step) * step for e in edges]
        
        edges = sorted(list(set(edges)))

        if len(edges) < 2:
            edges = [lower_bound, valid_max]

    # -----------------------------------
    # 7. MECE Rule 생성
    # -----------------------------------
    bins = []
    v_labels = {}
    num_segments = len(edges) - 1

    for i in range(num_segments):
        rule = {'valL': None, 'opL': '선택안함', 'valR': None, 'opR': '선택안함'}

        if i == 0:
            rule['valR'] = float(edges[1])
            rule['opR'] = '이하'
        elif i == num_segments - 1 and num_segments > 1:
            rule['valL'] = float(edges[-2])
            rule['opL'] = '초과'
            rule['valR'] = None
            rule['opR'] = '선택안함'
        else:
            rule['valL'] = float(edges[i])
            rule['opL'] = '초과'
            rule['valR'] = float(edges[i+1])
            rule['opR'] = '이하'

        bins.append(rule)

        lp = []
        is_discrete = not continuous
        # Left part
        if rule['opL'] == "이상":
            lp.append(f"{rule['valL']} 이상")
        elif rule['opL'] == "초과":
            if is_discrete:
                lp.append(f"{rule['valL'] + 1.0} 이상") # 1 초과 -> 2 이상
            else:
                lp.append(f"{rule['valL']} 초과")
        
        # Right part
        if rule['opR'] == "이하":
            lp.append(f"{rule['valR']} 이하")
        elif rule['opR'] == "미만":
            if is_discrete:
                lp.append(f"{rule['valR'] - 1.0} 이하") # 10 미만 -> 9 이하
            else:
                lp.append(f"{rule['valR']} 미만")
        
        v_labels[i+1] = " ".join(lp) if lp else "전체"

    return {
        "continuous": continuous,
        "k": len(bins),
        "bins": bins,
        "value_labels": v_labels,
        "spss_notes": {
            "epsilon_used": continuous,
            "epsilon_value": calculate_epsilon(series) if continuous else 1.0
        }
    }

def apply_rules(series, rule):
    """Applies dual-operator range rules sequentially."""
    res = pd.Series(np.nan, index=series.index)
    
    for i, bin_cfg in enumerate(rule.get('bins', [])):
        code = i + 1
        mask = pd.Series(True, index=series.index)
        
        # Left Bound
        vL, opL = bin_cfg.get('valL'), bin_cfg.get('opL', '선택안함')
        if opL == "이상" and vL is not None:
            mask &= (series >= vL)
        elif opL == "초과" and vL is not None:
            mask &= (series > vL)
            
        # Right Bound
        vR, opR = bin_cfg.get('valR'), bin_cfg.get('opR', '선택안함')
        if opR == "이하" and vR is not None:
            mask &= (series <= vR)
        elif opR == "미만" and vR is not None:
            mask &= (series < vR)
            
        # First-match wins
        res[mask & res.isna()] = code
        
    if rule.get('missing_policy') == 'recode':
        res[series.isna()] = rule.get('missing_code')
        
    return res

def build_spss_syntax(df, rules_dict):
    """Generates SPSS syntax based on rules and epsilon strategy."""
    syntax = []
    
    for var, rule in rules_dict.items():
        if rule.get('skip', False):
            continue
            
        new_var = rule['new_var']
        label = rule['label'].replace("'", "''") # Escape single quotes
        # Continuous: small epsilon, Discrete(Integer): eps = 1
        is_cont = rule.get('continuous', True)
        eps = rule.get('spss_notes', {}).get('epsilon_value', 1e-6 if is_cont else 1.0)
        
        recode_parts = []
        for i, bin_cfg in enumerate(rule.get('bins', [])):
            code = i + 1
            vL, opL = bin_cfg.get('valL'), bin_cfg.get('opL', '선택안함')
            vR, opR = bin_cfg.get('valR'), bin_cfg.get('opR', '선택안함')
            
            # Map to SPSS range keywords
            lo_str = "LOWEST"
            if opL == "이상" and vL is not None: lo_str = f"{vL}"
            elif opL == "초과" and vL is not None: lo_str = f"{vL + eps}"
            
            hi_str = "HIGHEST"
            if opR == "이하" and vR is not None: hi_str = f"{vR}"
            elif opR == "미만" and vR is not None: hi_str = f"{vR - eps}"
            
            if lo_str == "LOWEST" and hi_str == "HIGHEST":
                recode_parts.append(f"(ELSE = {code})")
            else:
                recode_parts.append(f"({lo_str} THRU {hi_str} = {code})")
                
        # Missing policy
        if rule.get('missing_policy') == 'recode':
            m_code = rule.get('missing_code')
            recode_parts.append(f"(SYSMIS = {m_code})")
            
        syntax.append(f"RECODE {var} {' '.join(recode_parts)} INTO {new_var}.")
        syntax.append(f"VARIABLE LABELS {new_var} '{label}_범주'.")
        syntax.append("EXECUTE.\n")
        
        # Value Labels
        v_labels = rule.get('value_labels', {})
        if v_labels:
            syntax.append(f"VALUE LABELS")
            syntax.append(f"/{new_var}")
            for code, v_label in v_labels.items():
                safe_v_label = str(v_label).replace("'", "''")
                syntax.append(f"{code} '{safe_v_label}'")
            syntax.append(".")
            syntax.append("EXECUTE.\n")
            
    header = "* Encoding: UTF-8.\n* SPSS Syntax generated by Data Categorization Agent.\n\n"
    return header + "\n".join(syntax)

def build_long_dataset_v2(df_resp, df_meta, mapping, set_key, banner_cols, id_col='idx'):
    """
    Metadata-template based Wide to Long transformation.
    Handles 'X' placeholder for person_no.
    Long keys: [idx, set_no/group_no, person_no].
    """
    if id_col not in df_resp.columns:
        if 'idx' in df_resp.columns: id_col = 'idx'
        else:
            df_resp = df_resp.reset_index().rename(columns={'index': '_id_tmp'})
            id_col = '_id_tmp'

    # 1. Identify Templates from Metadata
    # Q4: ^Q4_(\d+)_X_(.+)$ -> set_no, suffix
    # Q7: ^Q7_4_(\d+)_X_(.+)$ -> group_no, suffix
    templates = []
    if set_key == "Q4":
        pattern = r'^Q4_(\d+)_X_(.+)$'
    elif set_key == "Q7":
        pattern = r'^Q7_4_(\d+)_X_(.+)$'
    else:
        return None

    var_idx = mapping['var']
    for _, row in df_meta.iterrows():
        var_template = str(row.iloc[var_idx])
        match = re.match(pattern, var_template)
        if match:
            # group(1) = set_no or group_no, group(2) = suffix
            templates.append({'num': match.group(1), 'suffix': match.group(2)})
    
    if not templates:
        return None

    # Unique sets/groups
    set_nums = sorted(list(set(t['num'] for t in templates)))
    
    all_long_dfs = []
    
    for s_num in set_nums:
        # Skip Q4 etc-slots (6 and 7) which are not regular sets
        if set_key == "Q4" and str(s_num) in ["6", "7"]:
            continue
            
        # For this set/group, find all Suffixes
        suffixes = [t['suffix'] for t in templates if t['num'] == s_num]
        
        # Suffix filter: Drop etc
        suffixes = [s for s in suffixes if not re.search(r'(_etc$|_etc_|^etc$)', s, re.I)]
        
        # Find actual columns in df_resp for this s_num
        # Actual: e.g. Q4_{s_num}_{person_no}_{suffix}
        # We need to detect person_no range
        relevant_cols = []
        if set_key == "Q4":
            data_pattern = rf'^Q4_{s_num}_(\d+)_(.+)$'
        else: # Q7
            data_pattern = rf'^Q7_4_{s_num}_(\d+)_(.+)$'
            
        col_map = {} # (person_no, suffix) -> actual_col
        person_nums = set()
        
        for c in df_resp.columns:
            m = re.match(data_pattern, c)
            if m:
                p_no = m.group(1)
                sfx = m.group(2)
                if sfx in suffixes:
                    col_map[(p_no, sfx)] = c
                    person_nums.add(p_no)
        
        if not person_nums:
            continue
            
        long_rows = []
        sorted_persons = sorted(list(person_nums), key=int)
        
        for p_no in sorted_persons:
            # Check if this person block is all NA
            block_cols = [col_map[(p_no, sfx)] for sfx in suffixes if (p_no, sfx) in col_map]
            if not block_cols: continue
            
            # Sub-dataframe for this person across all respondents
            # Use unique column list to avoid ValueError if id_col is in banner_cols
            use_cols = list(dict.fromkeys([id_col] + banner_cols + block_cols))
            temp_df = df_resp[use_cols].copy()
            # Drop rows where all block_cols are NA
            temp_df = temp_df.dropna(subset=block_cols, how='all')
            
            if temp_df.empty: continue
            
            # Map block_cols to suffix names
            rename_map = {col_map[(p_no, sfx)]: sfx for sfx in suffixes if (p_no, sfx) in col_map}
            temp_df = temp_df.rename(columns=rename_map)
            
            # Add keys
            if set_key == "Q4":
                temp_df['set_no'] = int(s_num)
            else: # Q7
                temp_df['group_no'] = int(s_num)
            temp_df['person_no'] = int(p_no)
            
            long_rows.append(temp_df)
            
        if long_rows:
            all_long_dfs.append(pd.concat(long_rows, ignore_index=True))
            
    if not all_long_dfs:
        return None
        
    return pd.concat(all_long_dfs, ignore_index=True)

def compute_time_decimal_and_total_v2(df_long):
    """
    Computes start_S, end_S, T_S based on consistent suffix columns in Long DF.
    Expected suffixes: '4_1'(시), '4_2'(분) or '4'(시/분 통합 또는 시만 있음)
    """
    # 1. Start Time
    start_col = '4_1' if '4_1' in df_long.columns else ('4' if '4' in df_long.columns else None)
    if start_col:
        h = pd.to_numeric(df_long[start_col], errors='coerce')
        m_col = '4_2' if '4_2' in df_long.columns else None
        m = pd.to_numeric(df_long[m_col], errors='coerce').fillna(0) if m_col else pd.Series(0, index=df_long.index)
        df_long['start_S'] = h + (m / 60)
        df_long['flag_start_range'] = ((df_long['start_S'] < 0) | (df_long['start_S'] > 24.1)).astype(int)
    
    # 2. End Time
    end_col = '5_1' if '5_1' in df_long.columns else ('5' if '5' in df_long.columns else None)
    if end_col:
        h = pd.to_numeric(df_long[end_col], errors='coerce')
        m_col = '5_2' if '5_2' in df_long.columns else None
        m = pd.to_numeric(df_long[m_col], errors='coerce').fillna(0) if m_col else pd.Series(0, index=df_long.index)
        df_long['end_S'] = h + (m / 60)
        df_long['flag_end_range'] = ((df_long['end_S'] < 0) | (df_long['end_S'] > 24.1)).astype(int)

    # 3. Total Time (if both exist)
    if 'start_S' in df_long.columns and 'end_S' in df_long.columns:
        df_long['T_S'] = df_long['end_S'] - df_long['start_S']
        # Handle overnight (e.g. 23 to 01 = -22 -> should be 2 hours)
        mask = df_long['T_S'] < 0
        df_long.loc[mask, 'T_S'] = df_long.loc[mask, 'T_S'] + 24
        df_long['flag_total_negative'] = 0 # Fixed automatically
        
    return df_long

def compute_missingness_with_skip_patterns(df, skip_rules_text):
    """Calculates missingness considering skip patterns."""
    report = []
    
    # We'll treat the skip_rules_text as documentation for now, 
    # but we can implement basic logic for the requested presets.
    
    has_ufq3 = 'ufQ3' in df.columns
    
    for col in df.columns:
        total_missing_count = df[col].isna().sum()
        total_rate = total_missing_count / len(df)
        
        abnormal_missing_rate = total_rate
        
        # Simple heuristic for presets if ufQ3 exists
        if has_ufq3:
            if re.match(r"Q2_2|Q4", col):
                # Normal if ufQ3 is 1 or 2
                normal_mask = df['ufQ3'].isin([1, 2])
                abnormal_missing_count = df[~normal_mask][col].isna().sum()
                abnormal_missing_rate = abnormal_missing_count / len(df)
            elif re.match(r"Q6|Q7", col):
                # Normal if ufQ3 is 3 or 4
                normal_mask = df['ufQ3'].isin([3, 4])
                abnormal_missing_count = df[~normal_mask][col].isna().sum()
                abnormal_missing_rate = abnormal_missing_count / len(df)
        
        report.append({
            "Variable": col,
            "Total Missing (%)": round(total_rate * 100, 2),
            "Abnormal Missing (%)": round(abnormal_missing_rate * 100, 2),
            "Status": "OK" if abnormal_missing_rate < 0.05 else "Check"
        })
    return pd.DataFrame(report)

# ==========================================
# 2. Main App Logic
# ==========================================

def main():
    st.title("📊 Data Categorization Agent")
    
    # Main Data State
    df_resp_raw = None
    df_meta = None
    mapping = None

    # Main Tabs
    tab_preprocess, tab_categorize = st.tabs(["전처리 (구조변환/시간변환)", "범주화 (K구간/신텍스)"])

    # Sidebar for Config & Uploads
    with st.sidebar:
        st.header("1. 파일 업로드")
        resp_file = st.file_uploader("응답데이터 (EXCEL/SAV)", type=['xlsx', 'sav', 'csv'])
        meta_file = st.file_uploader("메타데이터 (EXCEL)", type=['xlsx'])
        
        st.divider()
        st.header("2. 열 매핑 (기본값: C, D, E, F, H)")
        var_col = st.text_input("변수명 열 (예: C)", "C")
        label_col = st.text_input("문항내용 열 (예: D)", "D")
        type2_col = st.text_input("변수유형 열 (예: E, 시간/인원)", "E")
        type_col = st.text_input("분석유형 열 (예: F, 배너/척도/범주 등)", "F")
        flag_col = st.text_input("범주화 필요 여부 열 (예: H)", "H")
        
        if meta_file:
            st.divider()
            st.header("3. 메타데이터 설정")
            try:
                xl = pd.ExcelFile(meta_file)
                sheet_name = st.selectbox("메타데이터 시트", xl.sheet_names)
                df_meta = pd.read_excel(meta_file, sheet_name=sheet_name)
                
                if 'column_guide_labels' not in st.session_state or st.session_state.get('_last_meta_file') != meta_file.name:
                    st.session_state.column_guide_labels = parse_column_guide_labels(xl)
                    st.session_state._last_meta_file = meta_file.name
                
                def col_to_idx(c): return ord(c.upper()) - ord('A')
                mapping = {
                    'var': col_to_idx(var_col),
                    'label': col_to_idx(label_col),
                    'type2': col_to_idx(type2_col), # 신규 '변수유형' 열 (시간/인원)
                    'type': col_to_idx(type_col),   # 기존 '유형' 열 (F열 이동, 분석유형)
                    'flag': col_to_idx(flag_col)    # '범주화필요' 여부 (H열 이동)
                }
            except Exception as e:
                st.error(f"메타데이터 로드 실패: {e}")

    if resp_file:
        df_resp_raw, _ = load_data_file(resp_file)

    # --- Sidebar data check ---
    files_uploaded = (resp_file is not None) and (meta_file is not None) and (df_meta is not None) and (df_resp_raw is not None)

    # ==========================================
    # TAB: Preprocessing
    # ==========================================
    with tab_preprocess:
        if not files_uploaded:
            st.info("👋 좌측 사이드바에서 **응답데이터**와 **메타데이터**를 모두 업로드해 주세요.")
            st.image("https://img.icons8.com/clouds/200/000000/upload.png")
        else:
            st.header("데이터 전처리 (Wide -> Long & 시간 수치화)")
            
            st.subheader("1. 배너 변수 및 ID 선택")
            all_cols = list(df_resp_raw.columns)
            id_col = st.selectbox("ID 열 (기관 식별자)", all_cols, index=all_cols.index('idx') if 'idx' in all_cols else 0)
            
            # Find default banners from metadata (Column E contains "배너")
            default_banners = []
            if df_meta is not None and mapping:
                type_idx = mapping['type']
                var_idx = mapping['var']
                # Filter rows where Type column contains '배너'
                mask = df_meta.iloc[:, type_idx].astype(str).str.contains("배너", na=False)
                meta_banners = df_meta[mask].iloc[:, var_idx].astype(str).tolist()
                default_banners = [v for v in meta_banners if v in all_cols]
            
            # Additional hardcoded defaults requested by user
            extra_defaults = ['ufQ1', 'ufQ2', 'DQ7_2', 'DQ7_1a', 'DQ8', 'DQ10']
            for col in extra_defaults:
                if col in all_cols and col not in default_banners:
                    default_banners.append(col)
            
            banner_cols = st.multiselect("배너 변수 선택 (결과에 유지할 컬럼)", all_cols, default=default_banners)
            
            st.subheader("2. 전처리 세트 선택")
            preprocess_sets = st.multiselect("구조변환 대상 세트 선택", ["Q4", "Q7"], default=[])
            
            if st.button("전처리 실행"):
                with st.spinner("전처리 중..."):
                    results = {}
                    for s_key in preprocess_sets:
                        try:
                            # Use V2: Handles X placeholder and etc-drop
                            df_res = build_long_dataset_v2(df_resp_raw, df_meta, mapping, s_key, banner_cols, id_col)
                            if df_res is not None and not df_res.empty:
                                # Apply Time Calculation to both Q4 and Q7
                                if s_key in ["Q4", "Q7"]:
                                    df_res = compute_time_decimal_and_total_v2(df_res)
                                
                                # Apply Renaming based on User Table
                                if s_key == "Q4":
                                    ren_map = {
                                        '1': 'Q4_K_X_1', '2': 'Q4_K_X_2', '3': 'Q4_K_X_3',
                                        'start_S': 'Q4_K_X_4_S', 'end_S': 'Q4_K_X_5_S', 'T_S': 'Q4_K_X_T_S'
                                    }
                                    df_res = df_res.rename(columns={k: v for k, v in ren_map.items() if k in df_res.columns})
                                elif s_key == "Q7":
                                    ren_map = {
                                        '1': 'Q7_4_J_X_1', '2': 'Q7_4_J_X_2', '3': 'Q7_4_J_X_3',
                                        'start_S': 'Q7_4_J_X_4_S', 'end_S': 'Q7_4_J_X_5_S', 'T_S': 'Q7_4_J_X_T_S'
                                    }
                                    df_res = df_res.rename(columns={k: v for k, v in ren_map.items() if k in df_res.columns})
                                    
                                results[s_key] = df_res
                        except Exception as e:
                            st.error(f"{s_key} 세트 처리 중 오류: {e}")
                            
                    if results:
                        st.session_state.processed_results = results
                        st.success(f"{len(results)}개 세트 전처리 완료!")
                    else:
                        st.warning("전처리 대상 데이터를 찾지 못했습니다. 메타데이터의 'X' 템플릿을 확인해주세요.")
            
            # Persistent results display
            if "processed_results" in st.session_state:
                results = st.session_state.processed_results
                st.divider()
                # Previews (Ordered)
                for s_key in sorted(results.keys()):
                    res_df = results[s_key]
                    st.write(f"### {s_key} 결과 미리보기")
                    st.write(f"- 행 수: **{len(res_df)}**")
                    st.write(f"- 유니크 대상자(person_no) 수: **{res_df['person_no'].nunique()}**")
                    
                    # Row count summary by set/group
                    group_col = 'set_no' if 'set_no' in res_df.columns else 'group_no'
                    if group_col in res_df.columns:
                        st.write("#### 세트별 행 수 요약")
                        summary_df = res_df.groupby(group_col).size().reset_index(name='행 수')
                        st.table(summary_df)

                    st.dataframe(res_df.head())
                    
                # Excel Creation (Outside UI loop)
                st.subheader("3. 결과 다운로드")
                out_filename = st.text_input("다운로드 파일명 설정", value="preprocessed_data", help="확장자(.xlsx 또는 .zip)는 자동으로 추가됩니다.", key="pre_fn_input")
                if not out_filename.strip(): out_filename = "preprocessed_data"
                
                # Checkbox for splitting by set_no / group_no
                split_download = st.checkbox("세트(set_no / group_no)별로 분할하여 다운로드 (ZIP 압축)", value=False)
                
                # Shared Variable Info Logic
                var_info_list = []
                def add_info(base_suffix, label):
                    for s_key in ["Q4", "Q7"]:
                        if s_key in results:
                            target_df = results[s_key]
                            prefix = "Q4_K_X_" if s_key == "Q4" else "Q7_4_J_X_"
                            renamed = f"{prefix}{base_suffix}"
                            if renamed in target_df.columns:
                                var_info_list.append([renamed, label])
                                return
                            if base_suffix in target_df.columns:
                                var_info_list.append([base_suffix, label])
                                return

                add_info("1", "구분")
                add_info("2", "주된 역할")
                add_info("3", "자격 기준")
                add_info("4_1", "시작 시각(시)")
                add_info("4_2", "시작 시각(분)")
                add_info("5_1", "종료 시각(시)")
                add_info("5_2", "종료 시각(분)")
                
                if "Q4" in results: var_info_list.append(["set_no", "서비스 종류 (Q4 전용)"])
                if "Q7" in results: var_info_list.append(["group_no", "담당 학급 (Q7 전용)"])
                    
                add_info("person_no", "인력 번호")
                add_info("4_S", "시작 시각(수치형)")
                add_info("5_S", "종료 시각(수치형)")
                add_info("T_S", "총 근무 시간(수치형)")
                
                var_info_list += [
                    ["flag_start_range", "시작 시각 이상 여부"],
                    ["flag_end_range", "종료 시각 이상 여부"],
                    ["flag_total_negative", "총 근무 시간 이상 여부"]
                ]
                df_info = pd.DataFrame(var_info_list, columns=["변수명", "레이블"])

                # Logic Branch: Split vs Single
                if split_download:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        # 1. Add Codebook info to Zip
                        info_buffer = io.BytesIO()
                        df_info.to_excel(info_buffer, index=False, engine='xlsxwriter')
                        zip_file.writestr("변수정보_코드북.xlsx", info_buffer.getvalue())
                        
                        # 2. Iterate and split dataframes
                        for s_key, df_res in results.items():
                            group_col = 'set_no' if 'set_no' in df_res.columns else ('group_no' if 'group_no' in df_res.columns else None)
                            
                            if group_col:
                                for g_val, group_df in df_res.groupby(group_col):
                                    g_val_str = str(int(g_val)) if pd.notna(g_val) else "NA"
                                    # Create individual Excel
                                    file_buffer = io.BytesIO()
                                    target_filename = f"{out_filename}_{s_key}_set{g_val_str}.xlsx"
                                    group_df.to_excel(file_buffer, index=False, engine='xlsxwriter')
                                    zip_file.writestr(target_filename, file_buffer.getvalue())
                            else:
                                # Fallback if no grouping column
                                file_buffer = io.BytesIO()
                                target_filename = f"{out_filename}_{s_key}.xlsx"
                                df_res.to_excel(file_buffer, index=False, engine='xlsxwriter')
                                zip_file.writestr(target_filename, file_buffer.getvalue())
                                
                    st.download_button(
                        label="세트별 분할 데이터 다운로드 (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"{out_filename}.zip",
                        mime="application/zip"
                    )
                else:
                    # Original Integrated Excel output
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        for s_key in sorted(results.keys()):
                            results[s_key].to_excel(writer, sheet_name=s_key, index=False)
                        df_info.to_excel(writer, sheet_name="변수정보", index=False)
                            
                    st.download_button("전처리 결과 데이터 다운로드 (엑셀)", data=buffer.getvalue(), file_name=f"{out_filename}.xlsx")

                # Store in session state for next tab
                st.session_state.results = results

    # ==========================================
    # TAB: Categorization
    # ==========================================
    with tab_categorize:
        if not files_uploaded:
            st.info("👋 좌측 사이드바에서 **응답데이터**와 **메타데이터**를 모두 업로드해 주세요.")
        else:
            # Filter "범주화 필요"
            try:
                max_idx = max(mapping.values())
                if df_meta.shape[1] <= max_idx:
                    st.error(f"메타데이터에 열이 부족합니다. (필요 인덱스: {max_idx}, 실제 열 수: {df_meta.shape[1]})")
                else:
                    meta_data = df_meta.iloc[:, [mapping['var'], mapping['label'], mapping['type2'], mapping['type'], mapping['flag']]]
                    meta_data.columns = ['var', 'label', 'type2', 'type', 'flag']
                    meta_data['flag'] = meta_data['flag'].astype(str).str.strip().str.lower()
                    target_vars = meta_data[meta_data['flag'].str.contains("범주화 필요", na=False)]
                    
                    if target_vars.empty:
                        st.warning("메타데이터에서 '범주화 필요'로 표시된 변수를 찾지 못했습니다.")
                    else:
                        # Data Selection
                        data_source = "원본 데이터"
                        source_options = ["원본 데이터"]
                        if 'results' in st.session_state and st.session_state.results:
                            source_options += list(st.session_state.results.keys())
                        source_options.append("새로 업로드")
                        data_source = st.radio("데이터 소스 선택", source_options, horizontal=True, key="cat_data_source")
                        
                        # --- 새로 업로드 처리 ---
                        cat_meta_override = None
                        if data_source == "새로 업로드":
                            st.divider()
                            st.markdown("**📂 범주화용 데이터 새로 업로드**")
                            uc1, uc2 = st.columns(2)
                            with uc1:
                                cat_resp_file = st.file_uploader("범주화 데이터 (EXCEL/SAV/CSV)", type=['xlsx', 'sav', 'csv'], key="cat_resp_upload")
                            with uc2:
                                cat_meta_file = st.file_uploader("범주화 메타데이터 (EXCEL)", type=['xlsx'], key="cat_meta_upload")
                            
                            if cat_resp_file is None:
                                st.info("👆 범주화할 데이터 파일을 업로드해 주세요.")
                                df_to_use = None
                            else:
                                df_to_use, _ = load_data_file(cat_resp_file)
                                st.success(f"데이터 로드 완료: {df_to_use.shape[0]}행 × {df_to_use.shape[1]}열")
                            
                            # Override meta if provided
                            if cat_meta_file:
                                try:
                                    xl_cat = pd.ExcelFile(cat_meta_file)
                                    cat_sheet = st.selectbox("메타데이터 시트 (새 업로드)", xl_cat.sheet_names, key="cat_meta_sheet")
                                    cat_meta_override = pd.read_excel(cat_meta_file, sheet_name=cat_sheet)
                                    st.session_state.cat_tab_guide_labels = parse_column_guide_labels(xl_cat)
                                    st.success("메타데이터 로드 완료")
                                except Exception as e:
                                    st.error(f"새 메타데이터 로드 실패: {e}")
                            
                            st.divider()
                            
                            if df_to_use is None:
                                st.stop()
                        else:
                            df_to_use = df_resp_raw
                            if data_source != "원본 데이터" and 'results' in st.session_state:
                                df_to_use = st.session_state.results[data_source]
                        
                        # 오버라이드된 메타 사용 (새 업로드 시)
                        active_meta = cat_meta_override if cat_meta_override is not None else df_meta
                        
                        # 새 메타가 있으면 target_vars 재계산
                        if cat_meta_override is not None:
                            try:
                                cat_meta_data = cat_meta_override.iloc[:, [mapping['var'], mapping['label'], mapping['type2'], mapping['type'], mapping['flag']]]
                                cat_meta_data.columns = ['var', 'label', 'type2', 'type', 'flag']
                                cat_meta_data['flag'] = cat_meta_data['flag'].astype(str).str.strip().str.lower()
                                target_vars = cat_meta_data[cat_meta_data['flag'].str.contains("범주화 필요", na=False)]
                            except Exception:
                                pass  # 기존 target_vars 유지

                        def on_load_p():
                            sel = st.session_state.preset_load_sel
                            if sel != "선택 안 함":
                                st.session_state.rules = copy.deepcopy(st.session_state.cat_presets[sel])
                                st.session_state.current_preset_name = sel
                                st.session_state.preset_load_sel = "선택 안 함"
                                st.session_state.cat_last_msg = f"✅ '{sel}' 프리셋을 불러왔습니다."
                                log_action("불러오기", f"프리셋: {sel}")

                        def on_save_p():
                            val = st.session_state.preset_save_input
                            if val.strip():
                                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                full_name = f"{val.strip()} ({now_str})"
                                st.session_state.cat_presets[full_name] = copy.deepcopy(st.session_state.rules)
                                st.session_state.current_preset_name = full_name
                                st.session_state.preset_save_input = ""
                                st.session_state.cat_last_msg = f"✅ '{full_name}' 프리셋을 저장했습니다."
                                save_presets_to_file(st.session_state.cat_presets)
                                log_action("저장", f"이름: {full_name}")

                        def on_del_p(p_name):
                            if p_name in st.session_state.cat_presets:
                                del st.session_state.cat_presets[p_name]
                                if st.session_state.get('current_preset_name') == p_name:
                                    st.session_state.current_preset_name = "없음 (방금 삭제됨)"
                                save_presets_to_file(st.session_state.cat_presets)
                                log_action("삭제", f"프리셋: {p_name}")

                        def on_import_spss():
                            up = st.session_state.spss_import_file
                            if up:
                                try:
                                    text = up.read().decode('utf-8-sig', errors='ignore')
                                    imported_rules = parse_spss_syntax(text)
                                    if imported_rules:
                                        st.session_state.rules = imported_rules
                                        st.session_state.current_preset_name = f"Imported ({up.name})"
                                        st.session_state.cat_last_msg = f"✅ '{up.name}'에서 규칙을 복원했습니다."
                                        log_action("신텍스 복원", f"파일명: {up.name}")
                                    else:
                                        st.error("신텍스에서 유효한 규칙을 찾지 못했습니다.")
                                except Exception as e:
                                    st.error(f"파싱 오류: {e}")

                        # Persistence initialization
                        if 'cat_presets' not in st.session_state:
                            st.session_state.cat_presets = load_presets_from_file()
                        if 'cat_audit_log' not in st.session_state:
                            st.session_state.cat_audit_log = load_audit_log_from_file()
                        if 'current_preset_name' not in st.session_state:
                            st.session_state.current_preset_name = "없음 (수동 설정 중)"
                        
                        with st.expander("📝 범주화 세팅 프리셋 관리", expanded=False):
                            st.markdown(f"🚩 **현재 적용된 프리셋:** `{st.session_state.current_preset_name}`")
                            if st.session_state.get('cat_last_msg'):
                                st.info(st.session_state.cat_last_msg)
                            
                            tab_p1, tab_p2 = st.tabs(["📂 프리셋 불러오기/저장", "📜 SPSS 신텍스로 복원"])
                            
                            with tab_p1:
                                pc1, pc2 = st.columns([3, 1])
                                preset_names = list(st.session_state.cat_presets.keys())
                                
                                with pc1:
                                    st.selectbox("불러올 프리셋 선택", ["선택 안 함"] + preset_names, key="preset_load_sel")
                                with pc2:
                                    st.write("") # padding
                                    st.write("")
                                    st.button("📥 불러오기", use_container_width=True, key="preset_load_btn", on_click=on_load_p)
                                        
                                sc1, sc2 = st.columns([3, 1])
                                with sc1:
                                    st.text_input("새 결과로 저장할 이름 입력", placeholder="예: 시나리오_A", key="preset_save_input")
                                with sc2:
                                    st.write("") # padding
                                    st.write("")
                                    st.button("💾 현재 세팅 저장", use_container_width=True, key="preset_save_btn", on_click=on_save_p)
                                
                                if preset_names:
                                    st.divider()
                                    st.markdown("**📋 저장된 프리셋 목록**")
                                    for p_name in sorted(preset_names, reverse=True):
                                        rc1, rc2 = st.columns([4, 1])
                                        rc1.text(f"• {p_name}")
                                        if rc2.button("삭제", key=f"del_indiv_{p_name}", use_container_width=True):
                                            on_del_p(p_name)
                                            st.rerun()
                                    
                                    st.divider()
                                    if st.button("🧨 모든 프리셋 초기화(삭제)", use_container_width=True, type="primary", key="preset_del_all_btn"):
                                        st.session_state.cat_presets = {}
                                        st.session_state.current_preset_name = "없음 (전체 삭제됨)"
                                        save_presets_to_file({})
                                        st.warning("모든 프리셋이 삭제되었습니다.")
                                        st.rerun()

                            with tab_p2:
                                st.info("기존에 생성했던 SPSS 신텍스(.sps) 파일을 업로드하여 범주화 설정을 복원할 수 있습니다.")
                                st.file_uploader("SPSS 신텍스 파일 업로드", type=['sps'], key="spss_import_file", on_change=on_import_spss)

                        inner_tabs = st.tabs(["📋 데이터 진단", "⚙️ 범주화 설정", "📝 감사 로그", "💾 다운로드"])
            
                        # --- Diagnostic ---
                        with inner_tabs[0]:
                            st.subheader(f"데이터 프리뷰 ({data_source})")
                            st.dataframe(df_to_use.head(10))
                            c1, c2 = st.columns(2)
                            c1.metric("총 행 수", df_to_use.shape[0])
                            c2.metric("총 열 수", df_to_use.shape[1])
                            missing_report = compute_missingness_with_skip_patterns(df_to_use, "Default")
                            st.dataframe(missing_report, use_container_width=True)
            
                        # --- Settings ---
                        # Algorithm versioning check: auto-reset if version mismatch
                        if st.session_state.get('cat_algo_version', 0) < CAT_ALGO_VERSION:
                            st.session_state.rules = {}
                            st.session_state.cat_algo_version = CAT_ALGO_VERSION
                            st.info("🔄 범주화 알고리즘이 업데이트되어 세팅이 초기화되었습니다.")

                        # 데이터 소스가 변경되었으면 rules 초기화
                        if st.session_state.get('_last_cat_source') != data_source:
                            st.session_state.rules = {}
                            st.session_state['_last_cat_source'] = data_source
                        if 'rules' not in st.session_state: st.session_state.rules = {}

                        with inner_tabs[1]:
                            for idx, row in target_vars.iterrows():
                                v_name, v_label = str(row['var']), str(row['label'])
                                # Skip if not in data OR entirely NA
                                if v_name not in df_to_use.columns: continue
                                series = df_to_use[v_name]
                                if series.dropna().empty: continue
                                
                                with st.expander(f"Variable: {v_name} ({v_label})"):
                                    # --- 1. Distribution View ---
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.write("**기술 통계 (Quartiles)**")
                                        st.write(series.describe().to_frame().T)
                                    with col2:
                                        # Custom Histogram with Cumulative %
                                        valid_s = series.dropna()
                                        if len(valid_s.unique()) <= 20:
                                            # Discrete
                                            counts = valid_s.value_counts().sort_index().reset_index()
                                            counts.columns = ['Value', 'Count']
                                            counts['Value_Str'] = counts['Value'].astype(str)
                                        else:
                                            # Continuous
                                            hist, bin_edges = np.histogram(valid_s, bins=min(20, len(valid_s.unique())))
                                            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                                            counts = pd.DataFrame({
                                                'Value': bin_centers,
                                                'Count': hist,
                                                'Value_Str': [f"{bin_edges[i]:.2f}~{bin_edges[i+1]:.2f}" for i in range(len(hist))]
                                            })
                                            
                                        counts['Cum_Pct'] = (counts['Count'].cumsum() / counts['Count'].sum() * 100).round(1)
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Bar(
                                            x=counts['Value_Str'], 
                                            y=counts['Count'], 
                                            name='Count',
                                            marker_color='#636EFA',
                                            hovertemplate="Count: %{y}<extra></extra>"
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=counts['Value_Str'], 
                                            y=counts['Cum_Pct'], 
                                            name='Cum %', 
                                            yaxis='y2',
                                            mode='lines+markers',
                                            marker_color='#EF553B',
                                            hovertemplate="Cum %: %{y}%<extra></extra>"
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"{v_name} 분포",
                                            height=250, 
                                            margin=dict(l=20, r=20, t=40, b=20),
                                            showlegend=False,
                                            xaxis=dict(title=v_label, type='category'),
                                            yaxis=dict(title='Count', side='left'),
                                            yaxis2=dict(title='Cum %', side='right', overlaying='y', range=[0, 105])
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                    # --- 2. Configuration ---
                                    is_cont = detect_continuous(series)
                                    eps = calculate_epsilon(series)
                                    
                                    # Load or Initialize Rule
                                    if v_name not in st.session_state.rules:
                                        guide_labels_dict = st.session_state.get('cat_tab_guide_labels')
                                        if guide_labels_dict is None:
                                            guide_labels_dict = st.session_state.get('column_guide_labels', {})
                                        predefined = guide_labels_dict.get(v_name)

                                        # Default to 3 bins, pass row context to suggest_bins for type detection
                                        rule = suggest_bins(series, is_cont, k=3, label=v_label, var_type_val=str(row['type2']), predefined_labels=predefined)
                                        if rule:
                                            rule.update({'var':v_name, 'new_var':f"{v_name}_C", 'label':v_label, 'missing_policy':'keep', 'missing_code':9})
                                            st.session_state.rules[v_name] = rule
                                            
                                    rule = st.session_state.rules.get(v_name)
                                    
                                    # Header and Reset Button
                                    c1, c2 = st.columns([3, 1])
                                    with c1:
                                        st.write("**범주 기준 및 레이블 설정**")
                                    with c2:
                                        if st.button("🔄 영역 초기화", key=f"reset_btn_{data_source}_{v_name}", use_container_width=True):
                                            if v_name in st.session_state.rules:
                                                del st.session_state.rules[v_name]
                                            st.rerun()
                                    
                                    rule = st.session_state.rules.get(v_name)
                                    if rule:
                                        # Use the bins provided by rule
                                        actual_k = len(rule['bins'])
                                        for i in range(actual_k):
                                            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.5, 1.2, 0.2, 1.2, 1.5, 0.5, 0.5])
                                            # Left Bound
                                            rule['bins'][i]['valL'] = c1.number_input(
                                                f"L-{i+1}", 
                                                value=float(rule['bins'][i]['valL']) if rule['bins'][i]['valL'] is not None else 0.0,
                                                key=f"valL_{data_source}_{v_name}_{actual_k}_{i}"
                                            )
                                            rule['bins'][i]['opL'] = c2.selectbox(
                                                f"OpL-{i+1}", 
                                                ["선택안함", "이상", "초과"], 
                                                index=["선택안함", "이상", "초과"].index(rule['bins'][i].get('opL', '선택안함')),
                                                key=f"opL_{data_source}_{v_name}_{actual_k}_{i}"
                                            )
                                            # Placeholder
                                            c3.write("**X**")
                                            # Right Bound
                                            rule['bins'][i]['opR'] = c4.selectbox(
                                                f"OpR-{i+1}", 
                                                ["선택안함", "이하", "미만"], 
                                                index=["선택안함", "이하", "미만"].index(rule['bins'][i].get('opR', '선택안함')),
                                                key=f"opR_{data_source}_{v_name}_{actual_k}_{i}"
                                            )
                                            rule['bins'][i]['valR'] = c5.number_input(
                                                f"R-{i+1}", 
                                                value=float(rule['bins'][i]['valR']) if rule['bins'][i]['valR'] is not None else 0.0,
                                                key=f"valR_{data_source}_{v_name}_{actual_k}_{i}"
                                            )
                                            
                                            # Add/Remove specific rows
                                            with c6:
                                                st.write("") # Alignment
                                                st.write("")
                                                if st.button("➕", key=f"add_{data_source}_{v_name}_{actual_k}_{i}", help="이 구간 아래에 새 구간 추가"):
                                                    new_bin = {'valL': None, 'opL': '선택안함', 'valR': None, 'opR': '선택안함'}
                                                    rule['bins'].insert(i + 1, new_bin)
                                                    rule['k'] = len(rule['bins'])
                                                    st.rerun()
                                            with c7:
                                                st.write("")
                                                st.write("")
                                                if st.button("🗑️", key=f"rem_{data_source}_{v_name}_{actual_k}_{i}", help="이 구간 삭제"):
                                                    if len(rule['bins']) > 1:
                                                        rule['bins'].pop(i)
                                                        rule['k'] = len(rule['bins'])
                                                        st.rerun()
                                            
                                            # Compute new default label (Symbolic Notation -> Reverted to Korean + Discrete Optimization)
                                            is_d = not rule.get('continuous', True)
                                            new_parts = []
                                            if rule['bins'][i]['opL'] == "이상": 
                                                new_parts.append(f"{rule['bins'][i]['valL']} 이상")
                                            elif rule['bins'][i]['opL'] == "초과":
                                                if is_d: new_parts.append(f"{rule['bins'][i]['valL'] + 1.0} 이상")
                                                else: new_parts.append(f"{rule['bins'][i]['valL']} 초과")
                                            
                                            if rule['bins'][i]['opR'] == "이하": 
                                                new_parts.append(f"{rule['bins'][i]['valR']} 이하")
                                            elif rule['bins'][i]['opR'] == "미만":
                                                if is_d: new_parts.append(f"{rule['bins'][i]['valR'] - 1.0} 이하")
                                                else: new_parts.append(f"{rule['bins'][i]['valR']} 미만")
                                            
                                            new_default_lab = " ".join(new_parts) if new_parts else "전체"
                                            rule['value_labels'][i+1] = new_default_lab
                                            
                                        # Recalculate value labels indexing to be clean 1...K
                                        new_v_labels = {}
                                        is_d_global = not rule.get('continuous', True)
                                        for idx, b in enumerate(rule['bins']):
                                            lp = []
                                            if b['opL'] == "이상": lp.append(f"{b['valL']} 이상")
                                            elif b['opL'] == "초과":
                                                if is_d_global: lp.append(f"{b['valL'] + 1.0} 이상")
                                                else: lp.append(f"{b['valL']} 초과")
                                            
                                            if b['opR'] == "이하": lp.append(f"{b['valR']} 이하")
                                            elif b['opR'] == "미만":
                                                if is_d_global: lp.append(f"{b['valR'] - 1.0} 이하")
                                                else: lp.append(f"{b['valR']} 미만")
                                            
                                            new_v_labels[idx+1] = " ".join(lp) if lp else "전체"
                                        rule['value_labels'] = new_v_labels
                                            

                                        
                                        # --- 3. Distribution Summary (Automatic) ---
                                        st.divider()
                                        st.write("📊 **범주별 분포 요약 (자동 갱신)**")
                                        
                                        try:
                                            # Calculate independent matching counts for each rule to show overlaps
                                            counts_data = []
                                            for i, bin_cfg in enumerate(rule.get('bins', [])):
                                                code = i + 1
                                                mask = pd.Series(True, index=series.index)
                                                
                                                vL, opL = bin_cfg.get('valL'), bin_cfg.get('opL', '선택안함')
                                                if opL == "이상" and vL is not None: mask &= (series >= vL)
                                                elif opL == "초과" and vL is not None: mask &= (series > vL)
                                                    
                                                vR, opR = bin_cfg.get('valR'), bin_cfg.get('opR', '선택안함')
                                                if opR == "이하" and vR is not None: mask &= (series <= vR)
                                                elif opR == "미만" and vR is not None: mask &= (series < vR)
                                                
                                                counts_data.append({'Code': code, 'Count': mask.sum()})
                                                
                                            counts = pd.DataFrame(counts_data)
                                            
                                            valid_n = len(series.dropna())
                                            counts['Percent'] = (counts['Count'] / valid_n * 100).round(1)
                                            
                                            # Map range info from structured bins
                                            intervals = {}
                                            for i, b in enumerate(rule['bins']):
                                                parts = []
                                                if b.get('opL', '선택안함') != '선택안함': parts.append(f"{b.get('valL')} {b.get('opL')}")
                                                if b.get('opR', '선택안함') != '선택안함': parts.append(f"{b.get('valR')} {b.get('opR')}")
                                                intervals[i+1] = " ".join(parts) if parts else "전체"
                                            counts['Range'] = counts['Code'].map(intervals)
                                            
                                            # Sort by Code ascending
                                            counts = counts.sort_values('Code')
                                            
                                            # Add Totals row
                                            total_row = pd.DataFrame([{
                                                'Code': '합계',
                                                'Range': '',
                                                'Count': f"{counts['Count'].sum()} / {valid_n}",
                                                'Percent': (counts['Count'].sum() / valid_n * 100).round(1) if valid_n > 0 else 0.0
                                            }])
                                            counts = pd.concat([counts, total_row], ignore_index=True)
                                            
                                            st.table(counts[['Code', 'Range', 'Count', 'Percent']])
                                            
                                            # Alert user if there are overlaps
                                            if counts['Count'][:-1].sum() > valid_n:
                                                st.warning("⚠️ **주의**: 설정하신 범위들이 서로 겹칩니다. (합계가 전체 데이터를 초과함) 실제 변환 및 SPSS에서는 **첫 번째로 일치하는 범주(위에서부터)** 에 우선 할당됩니다.")
                                        except Exception as e:
                                            st.error(f"분포 요약 생성 시 오류: {e}")

                        # --- Audit Log Tab ---
                        with inner_tabs[2]:
                            st.subheader("📋 범주화 작업 감사 로그 (Audit Log)")
                            st.info("프리셋 저장, 불러오기, 신텍스 복원 등의 이력이 최신순으로 표시됩니다. (최대 100건)")
                            if st.session_state.cat_audit_log:
                                st.table(pd.DataFrame(st.session_state.cat_audit_log))
                                if st.button("🧨 로그 전체 삭제", key="clear_audit_btn"):
                                    st.session_state.cat_audit_log = []
                                    save_audit_log_to_file([])
                                    st.rerun()
                            else:
                                st.write("기록된 작업 이력이 없습니다.")
            
                        # --- Export ---
                        with inner_tabs[3]:
                            st.subheader("결과 다운로드")
                            c1, c2 = st.columns(2)
                            
                            with c1:
                                st.info("1. 범주화 규칙이 적용된 전체 데이터를 엑셀로 내려받습니다.")
                                out_fn_cat = st.text_input("다운로드 파일명 설정", value=f"categorized_{data_source}", key="fn_cat_input")
                                if not out_fn_cat.strip(): out_fn_cat = f"categorized_{data_source}"
                                
                                if st.button("📊 범주화 결과 데이터 다운로드 (Excel)", key="final_exec_excel"):
                                    df_final = df_to_use.copy()
                                    processed_vars = set()
                                    
                                    # [Sync] Filter rules by current target_vars to exclude deleted/modified variables
                                    current_target_names = set(target_vars['var'].astype(str).tolist())
                                    
                                    for v, r in st.session_state.rules.items():
                                        # Only process if variable is in current metadata AND data
                                        if v in current_target_names and v in df_final.columns and not r.get('skip'):
                                            df_final[r['new_var']] = apply_rules(df_final[v], r)
                                            processed_vars.add(v)
                                    buf = io.BytesIO()
                                    df_final.to_excel(buf, index=False, engine='xlsxwriter')
                                    st.download_button("📥 데이터 (Excel) 파일 받기", buf.getvalue(), f"{out_fn_cat}.xlsx", key="dl_excel")
                                    
                            with c2:
                                st.info("2. 설정한 범주화 규칙을 SPSS 명령문(Syntax) 파일로 내려받습니다.")
                                if st.button("📜 SPSS 신텍스 생성 및 다운로드 (.sps)", key="final_exec_syntax"):
                                    # [Sync] Filter rules by current target_vars for SPSS syntax too
                                    current_target_names = set(target_vars['var'].astype(str).tolist())
                                    filtered_rules = {v: r for v, r in st.session_state.rules.items() if v in current_target_names}
                                    
                                    syntax = build_spss_syntax(df_to_use, filtered_rules)
                                    # UTF-8 BOM 인코딩으로 변환 (한글 깨짐 방지)
                                    syntax_bytes = syntax.encode('utf-8-sig')
                                    st.download_button("📥 SPSS (.sps) 파일 받기", syntax_bytes, "syntax.sps", mime="text/plain", key="dl_sps")
            except Exception as e:
                st.error(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
