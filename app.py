# app.py — Симулятор сценариев проекта «Микшер» (UI)

import os
import sys
import threading

# --- ИНТЕГРАЦИЯ M-Vave SMC-Mixer ---
# Импорт опционален: без установленных пакетов приложение работает как обычно (без пульта).
# УЛУЧШЕНИЕ: пульту нужен только mido (+python-rtmidi). Пакет streamlit_autorefresh не
# используется (фрагмент обновляется сам через run_every) — его отсутствие раньше ошибочно
# выключало весь MIDI.
try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

# В «оконном» (собранном без консоли) режиме у процесса нет stdout/stderr — они None,
# и Streamlit/click падают при попытке в них писать. Перенаправляем вывод в лог рядом с exe,
# ДО импорта streamlit. В обычном запуске (stdout есть) — ничего не делаем.
if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
    try:
        _logdir = os.path.dirname(sys.executable)
        _logf = open(os.path.join(_logdir, "mixer_run.log"), "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = _logf
        if sys.stderr is None:
            sys.stderr = _logf
    except Exception:
        pass

import io
import json
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import math
import textwrap
from datetime import datetime, timedelta

import mixer_core as mc
from mixer_core import ProjectMixer, LocalLLMEngine, DataLoaderOrchestrator, MixerConfig, RELATION_LABELS, parse_ru_number
import project_store as ps

# ======================================================================
# КОНФИГУРАЦИЯ
# ======================================================================
st.set_page_config(
    page_title="Микшер — сценарии проекта",
    page_icon="🎛️",
    layout="wide",
)

# 1. НЕСГОРАЕМАЯ ПАМЯТЬ ПУЛЬТА (переживает rerun'ы Streamlit)
@st.cache_resource
def get_midi_state():
    return {
        'mapping': [],      # Список привязанных сигналов
        'cc_modes': {},     # Тип ручки: 'relative' или 'absolute'
        'base': 100, 'req': 100, 'add': 0,
        'shift': 0, 'dur': 100,
        'trans_req': 0, 'trans_add': 0, # Освободили каналы 7 и 8 под перенос средств
        'updated': False,
        'revision': 0,      # монотонный счётчик изменений
        'port': None,       # имя подключённого порта
        'events': [],       # последние события MIDI (до 50 шт.)
    }


def _midi_log(msg: str):
    """Печать из фонового потока, безопасная для Windows-консоли (cp1251) и frozen-режима."""
    try:
        print(msg)
    except Exception:
        try:
            print(str(msg).encode('ascii', 'replace').decode())
        except Exception:
            pass
    # Также сохраняем событие в midi_state для UI
    try:
        midi_state = get_midi_state()
        evts = midi_state.get('events', [])
        evts.append(msg)
        midi_state['events'] = evts[-50:]  # максимум 50 последних событий
    except Exception:
        pass


@st.cache_resource
def start_midi_listener():
    if not MIDI_AVAILABLE:
        return None

    midi_state = get_midi_state()

    # Каналы пульта: (ключ, min, max, шаг чувствительности)
    CHANNELS = [
        ('base', 0, 100, 1),
        ('req', 0, 100, 1),
        ('add', 0, 100, 1),
        ('shift', -12, 24, 1),
        ('trans_req', 0, 100, 1), # Канал 7: Перенос потребности -> Базу (%)
        ('trans_add', 0, 100, 1), # Канал 8: Перенос доп. потребности -> Базу (%)
    ]

    def _find_port():
        try:
            ports = mido.get_input_names()
        except Exception:
            return None
        name = next((p for p in ports if 'SMC' in p or 'Vave' in p), None)
        if not name:
            name = next((p for p in ports if 'MIDI' in p), None)
        if not name and ports:
            name = ports[0]
        return name

    def midi_loop():
        import time
        while True:
            port_name = _find_port()
            if not port_name:
                midi_state['port'] = None
                time.sleep(3.0)
                continue
            try:
                _midi_log(f"✅ Подключен MIDI-пульт: {port_name}")
                midi_state['port'] = port_name
                with mido.open_input(port_name) as inport:
                    for msg in inport:
                        val = None
                        sig_id = None

                        if msg.type == 'control_change':
                            val = msg.value
                            sig_id = f"cc_{msg.control}"
                        elif msg.type == 'pitchwheel':
                            val = int((msg.pitch + 8192) / 16383.0 * 127)
                            sig_id = f"pw_{msg.channel}"

                        if val is None or sig_id is None:
                            continue

                        if sig_id not in midi_state['mapping']:
                            if len(midi_state['mapping']) < len(CHANNELS):
                                midi_state['mapping'].append(sig_id)
                                if msg.type == 'control_change' and val in (1, 2, 3, 4, 5, 6, 7,
                                                                            127, 126, 125, 124, 123, 122, 121,
                                                                            65, 66, 67, 63, 62, 61):
                                    midi_state['cc_modes'][sig_id] = 'relative'
                                    _midi_log(f"🎓 Привязана КРУТИЛКА [{sig_id}] к каналу {len(midi_state['mapping'])}")
                                else:
                                    midi_state['cc_modes'][sig_id] = 'absolute'
                                    _midi_log(f"🎓 Привязан ФЕЙДЕР [{sig_id}] к каналу {len(midi_state['mapping'])}")

                        if sig_id in midi_state['mapping']:
                            idx = midi_state['mapping'].index(sig_id)
                            key, min_v, max_v, step = CHANNELS[idx]
                            mode = midi_state['cc_modes'].get(sig_id, 'absolute')

                            if mode == 'relative':
                                delta = 0
                                if val in (1, 2, 3, 4, 5, 6, 7, 8): delta = val
                                elif val in (127, 126, 125, 124, 123, 122, 121, 120): delta = val - 128
                                elif val in (65, 66, 67, 68, 69): delta = val - 64
                                elif val in (63, 62, 61, 60, 59): delta = val - 64
                                new_v = midi_state[key] + (delta * step)
                            else:
                                new_v = min_v + (val / 127.0) * (max_v - min_v)

                            midi_state[key] = max(min_v, min(max_v, int(new_v)))
                            midi_state['updated'] = True
                            midi_state['revision'] = midi_state.get('revision', 0) + 1
            except Exception as e:
                _midi_log(f"❌ Ошибка MIDI: {e} — переподключение через 3 с")
                midi_state['port'] = None
                time.sleep(3.0)

    t = threading.Thread(target=midi_loop, daemon=True)
    t.start()
    return t

# Запускаем слушатель один раз при старте приложения
start_midi_listener()

PROJECTS_ROOT = ps.PROJECTS_ROOT

CHANGE_THRESHOLD = 1e-4  # ниже этого изменение KPI считаем «поглощённым» порогами активации


# ======================================================================
# ЛОКАЛЬНЫЕ РЕСУРСЫ (assets/) — для работы БЕЗ интернета
# ======================================================================
def _asset_dirs() -> list:
    """Каталоги поиска ресурсов assets/: рядом с exe, в бандле PyInstaller (_MEIPASS),
    рядом с исходником."""
    import sys as _sys
    dirs = []
    if getattr(_sys, "frozen", False):
        dirs.append(os.path.dirname(_sys.executable))
    dirs.append(getattr(_sys, "_MEIPASS", ""))
    dirs.append(os.path.dirname(os.path.abspath(__file__)))
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out


def _read_asset(*relparts) -> str:
    """Текстовый ресурс из assets/ (первый найденный) или пустая строка."""
    for base in _asset_dirs():
        p = os.path.join(base, *relparts)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return ""


def _read_asset_bytes(*relparts):
    """Бинарный ресурс из assets/ (первый найденный) или None."""
    for base in _asset_dirs():
        p = os.path.join(base, *relparts)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                pass
    return None


def _local_fonts_css() -> str:
    """Встраивает локальные шрифты из assets/fonts/ через @font-face (base64), чтобы НЕ
    качать их из интернета. Имя файла задаёт семейство и насыщенность: «Семейство-700.woff2»
    (напр. Inter-400.woff2, SpaceGrotesk-600.woff2, JetBrainsMono-600.woff2). Если файлов нет —
    вернёт пусто (интерфейс использует системные шрифты-фолбэки)."""
    import base64, glob
    fam_map = {  # префикс файла → CSS-семейство (как в переменных --mx-*)
        'inter': 'Inter', 'spacegrotesk': 'Space Grotesk', 'space-grotesk': 'Space Grotesk',
        'jetbrainsmono': 'JetBrains Mono', 'jetbrains-mono': 'JetBrains Mono', 'jetbrains': 'JetBrains Mono',
    }
    fmt = {'.woff2': 'woff2', '.woff': 'woff', '.ttf': 'truetype', '.otf': 'opentype'}
    faces = []
    fdir = None
    for base in _asset_dirs():
        cand = os.path.join(base, 'assets', 'fonts')
        if os.path.isdir(cand):
            fdir = cand; break
    if not fdir:
        return ""
    for path in sorted(glob.glob(os.path.join(fdir, '*'))):
        ext = os.path.splitext(path)[1].lower()
        if ext not in fmt:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.replace('_', '-').rsplit('-', 1)
        key = parts[0].lower().replace(' ', '')
        weight = parts[1] if len(parts) == 2 and parts[1].isdigit() else '400'
        family = fam_map.get(key)
        if not family:
            continue
        data = _read_asset_bytes('assets', 'fonts', os.path.basename(path))
        if not data:
            continue
        b64 = base64.b64encode(data).decode('ascii')
        faces.append(
            f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
            f"font-display:swap;src:url(data:font/{fmt[ext]};base64,{b64}) format('{fmt[ext]}');}}"
        )
    return ("<style>" + "".join(faces) + "</style>") if faces else ""


# Локальные шрифты (если положены в assets/fonts/) — вставляем ДО основного CSS.
_fcss = _local_fonts_css()
if _fcss:
    st.markdown(_fcss, unsafe_allow_html=True)

# ======================================================================
# ОФОРМЛЕНИЕ
# ======================================================================
st.markdown("""
<style>
:root{
  --mx-ink:#141922; --mx-ink-2:#374050; --mx-muted:#69727F; --mx-faint:#6E7682;
  --mx-surface:#FFFFFF; --mx-surface-2:#F5F7FA; --mx-bg:#ECEFF3;
  --mx-border:#E2E7EE; --mx-border-2:#D3DAE3;
  --mx-accent:#1B4DFF; --mx-accent-2:#2D6BFF; --mx-accent-soft:#EAEFFF; --mx-accent-ring:rgba(27,77,255,.18);
  --mx-live:#E0851C; --mx-live-soft:#FBEAD2;
  --mx-pos:#0E8F5E; --mx-neg:#D8425A; --mx-warn:#B26A08;
  --mx-pos-bg:#E2F4EC; --mx-neg-bg:#FBE7EB; --mx-warn-bg:#FBF0DC;
  --mx-shadow:0 1px 2px rgba(20,28,46,.05), 0 1px 3px rgba(20,28,46,.04);
  --mx-shadow-lift:0 10px 30px rgba(20,28,46,.10);
  --mx-r:14px; --mx-r-sm:10px;
  --mx-mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --mx-display:'Space Grotesk', 'Inter', system-ui, sans-serif;
  --mx-body:'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
html, body, .stApp, [class*="css"] { font-family: var(--mx-body); }
.stApp { background:
   radial-gradient(1200px 600px at 100% -10%, rgba(27,77,255,.05), transparent 60%),
   var(--mx-bg); color: var(--mx-ink); }
.block-container { padding-top: 1.4rem; padding-bottom: 3.2rem; max-width: 1580px; }
h1,h2,h3,h4 { font-family: var(--mx-display); color: var(--mx-ink); letter-spacing:-.02em; font-weight:600; }
h1 { font-size:1.7rem; }
a { color: var(--mx-accent); }
.mx-sub { color: var(--mx-muted); font-size: .95rem; margin:-.2rem 0 1rem; max-width:70ch; line-height:1.5; }
::selection { background: var(--mx-accent-ring); }

/* Заголовок секции: «сигнальная» метка-эйброу + название */
.mx-h { display:flex; align-items:baseline; gap:10px; margin:6px 0 10px; padding-bottom:8px;
        border-bottom:1px solid var(--mx-border); }
.mx-h::before{ content:""; align-self:center; width:9px; height:9px; border-radius:3px;
        background:linear-gradient(140deg,var(--mx-accent),var(--mx-accent-2)); box-shadow:0 0 0 4px var(--mx-accent-soft); }
.mx-h .t { font-family:var(--mx-display); font-weight:600; font-size:1.06rem; color:var(--mx-ink); letter-spacing:-.01em; }
.mx-h .s { color:var(--mx-faint); font-size:.78rem; font-weight:500; font-family:var(--mx-mono); text-transform:lowercase; }

/* Карточки */
.mx-card { background: var(--mx-surface); border:1px solid var(--mx-border); border-radius: var(--mx-r);
           padding:16px 18px; height:100%; box-shadow: var(--mx-shadow);
           transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease; }
.mx-card:hover { box-shadow: var(--mx-shadow-lift); transform: translateY(-2px); border-color: var(--mx-border-2); }
.mx-card .lbl { color:var(--mx-faint); font-size:.68rem; text-transform:uppercase; letter-spacing:.09em;
                font-weight:700; font-family:var(--mx-mono); }
.mx-card .name { font-weight:600; color:var(--mx-ink); }
.mx-card .plan { color:var(--mx-muted); font-size:.88rem; margin-top:3px; }

/* KPI как канал пульта: левый «сигнальный» торец + крупное моно-показание */
.mx-kpi { position:relative; overflow:hidden; padding-left:20px; }
.mx-kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
                  background:linear-gradient(180deg,var(--mx-accent),var(--mx-accent-2)); }
.mx-kpi.up::before  { background:linear-gradient(180deg,#16B277,var(--mx-pos)); }
.mx-kpi.down::before{ background:linear-gradient(180deg,#FF6A80,var(--mx-neg)); }
.mx-kpi .name { font-size:.95rem; margin:4px 0 8px; line-height:1.3; min-height:2.5em;
                display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.mx-kpi .row { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.mx-kpi .big { font-family:var(--mx-display); font-size:1.7rem; font-weight:700; color:var(--mx-ink);
               line-height:1; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.mx-kpi .dlt { font-size:.8rem; font-weight:600; color:var(--mx-muted); font-family:var(--mx-mono); margin-top:6px; }
/* Компактные строки по годам (многолетний проект) */
.mx-years { margin-top:10px; border-top:1px dashed var(--mx-border); padding-top:7px; display:flex; flex-direction:column; gap:3px; }
.mx-years .yr { display:flex; align-items:baseline; justify-content:space-between; gap:8px;
                font-family:var(--mx-mono); font-size:.78rem; font-variant-numeric:tabular-nums; }
.mx-years .yr .y { color:var(--mx-faint); font-weight:700; }
.mx-years .yr .pf { color:var(--mx-muted); }
.mx-years .yr .pf b { color:var(--mx-ink); font-weight:700; }
.mx-years .yr .pf b.pos { color:var(--mx-pos); } .mx-years .yr .pf b.neg { color:var(--mx-neg); }

/* Бейджи / чипы / пилюли */
.mx-badge { display:inline-block; padding:2px 9px; border-radius:7px; font-size:.8rem; font-weight:700;
            white-space:nowrap; font-family:var(--mx-mono); font-variant-numeric:tabular-nums; }
.mx-badge.pos { background:var(--mx-pos-bg); color:var(--mx-pos); }
.mx-badge.neg { background:var(--mx-neg-bg); color:var(--mx-neg); }
.mx-badge.flat{ background:var(--mx-surface-2); color:var(--mx-muted); }
.mx-pill { display:inline-block; padding:2px 9px; border-radius:6px; font-size:.74rem; font-weight:600;
           background:var(--mx-accent-soft); color:var(--mx-accent); margin:1px 3px 1px 0; }
.mx-tag-leaf   { background:#E7F0FF; color:#1556C9; }
.mx-tag-parent { background:#FBEAD2; color:#9A5800; }

.mx-ribbon { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:4px 0; }
.mx-chip { display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:9px;
           font-size:.84rem; font-weight:600; border:1px solid var(--mx-border); background:var(--mx-surface); }
.mx-chip .dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }
.mx-chip.ok  { background:var(--mx-pos-bg); color:var(--mx-pos); border-color:transparent; } .mx-chip.ok .dot{ background:var(--mx-pos);}
.mx-chip.warn{ background:var(--mx-warn-bg); color:var(--mx-warn); border-color:transparent; } .mx-chip.warn .dot{ background:var(--mx-warn);}
.mx-chip.bad { background:var(--mx-neg-bg); color:var(--mx-neg); border-color:transparent; } .mx-chip.bad .dot{ background:var(--mx-neg);}
.mx-chip.info{ background:var(--mx-accent-soft); color:var(--mx-accent); border-color:transparent; } .mx-chip.info .dot{ background:var(--mx-accent);}

/* Фейдер-метр: общая шкала, дорожка плана и уровень прогноза с «головкой» */
.mx-meter { margin-top:12px; }
.mx-meter .bar { height:8px; border-radius:6px; background:var(--mx-surface-2);
                 box-shadow:inset 0 1px 2px rgba(20,28,46,.06); position:relative; margin:4px 0; overflow:hidden; }
.mx-meter .fill { position:absolute; left:0; top:0; bottom:0; border-radius:6px; }
.mx-meter .fill.plan { background:repeating-linear-gradient(90deg,#C7CEDA,#C7CEDA 6px,#D5DBE5 6px,#D5DBE5 12px); }
.mx-meter .fill.pos  { background:linear-gradient(90deg,#16B277,var(--mx-pos)); }
.mx-meter .fill.neg  { background:linear-gradient(90deg,#FF6A80,var(--mx-neg)); }
.mx-meter .fill.acc  { background:linear-gradient(90deg,var(--mx-accent-2),var(--mx-accent)); }
.mx-meter .cap { display:flex; justify-content:space-between; font-size:.72rem; color:var(--mx-muted);
                 font-family:var(--mx-mono); font-variant-numeric:tabular-nums; }
.mx-meter .tick { position:absolute; top:-2px; bottom:-2px; width:2px; border-radius:2px;
                  background:var(--mx-ink); opacity:.5; transform:translateX(-1px); }

.mx-legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.76rem; color:var(--mx-muted); margin:4px 0 10px; }
.mx-legend span{ display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
.mx-legend i{ width:11px; height:11px; border-radius:3px; display:inline-block; }

.mx-diff { display:inline-flex; align-items:center; gap:7px; font-size:.86rem; font-family:var(--mx-mono); }
.mx-diff .was{ color:var(--mx-faint); text-decoration:line-through; }
.mx-diff .arr{ color:var(--mx-accent); font-weight:700; }
.mx-diff .now{ color:var(--mx-ink); font-weight:700; }

/* Кнопки — основная «сигнальная», вторичные тихие */
.stButton>button, .stDownloadButton>button {
   border-radius:10px; font-weight:600; font-family:var(--mx-body); border:1px solid var(--mx-border-2);
   background:var(--mx-surface); color:var(--mx-ink); transition:all .15s ease; padding:.45rem .9rem; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:var(--mx-accent); color:var(--mx-accent);
   box-shadow:0 4px 14px var(--mx-accent-ring); transform:translateY(-1px); }
.stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"] {
   background:linear-gradient(135deg,var(--mx-accent),var(--mx-accent-2)); color:#fff; border-color:transparent;
   box-shadow:0 6px 18px var(--mx-accent-ring); }
.stButton>button[kind="primary"]:hover { filter:brightness(1.05); color:#fff; transform:translateY(-1px); }
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline:2px solid var(--mx-accent); outline-offset:2px; }

/* Поля ввода как контролы пульта */
[data-baseweb="input"], [data-baseweb="select"]>div, .stNumberInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="input"], .stDateInput div[data-baseweb="input"] {
   border-radius:10px !important; border-color:var(--mx-border-2) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]>div:focus-within {
   border-color:var(--mx-accent) !important; box-shadow:0 0 0 3px var(--mx-accent-ring) !important; }
.stNumberInput input, .stTextInput input { font-family:var(--mx-mono); font-variant-numeric:tabular-nums; }

/* Слайдеры */
.stSlider [data-baseweb="slider"] [role="slider"]{ background:var(--mx-accent); box-shadow:0 0 0 4px var(--mx-accent-ring); }
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"]{ background:transparent; }
/* Клавиатурная доступность: видимый фокус на всех интерактивных контролах */
.stSlider [role="slider"]:focus-visible{ outline:3px solid var(--mx-accent); outline-offset:3px; }
.stCheckbox:focus-within label span:first-child,
.stRadio:focus-within label span:first-child{ box-shadow:0 0 0 3px var(--mx-accent-ring); border-radius:5px; }
[data-baseweb="tab"]:focus-visible{ outline:2px solid var(--mx-accent); outline-offset:-2px; border-radius:6px; }
[data-testid="stDataFrame"] [role="row"]:focus-visible,
a:focus-visible{ outline:2px solid var(--mx-accent); outline-offset:2px; }

/* Вкладки — подчёркивание + моно-метки */
.stTabs [data-baseweb="tab-list"]{ gap:2px; border-bottom:1px solid var(--mx-border); }
.stTabs [data-baseweb="tab"]{ font-weight:600; letter-spacing:.01em; color:var(--mx-muted); padding:8px 14px; }
.stTabs [aria-selected="true"]{ color:var(--mx-accent) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--mx-accent); height:2.5px; border-radius:2px; }

/* Таблицы, прогресс, экспандеры */
.stDataFrame, [data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid var(--mx-border); }
[data-testid="stProgress"] div[role="progressbar"]>div, .stProgress>div>div>div {
   background:linear-gradient(90deg,var(--mx-accent-2),var(--mx-accent)) !important; }
[data-testid="stExpander"], details { border:1px solid var(--mx-border) !important; border-radius:12px !important; background:var(--mx-surface); }
[data-testid="stMetricValue"]{ font-family:var(--mx-display); font-variant-numeric:tabular-nums; }
[data-testid="stMetricDelta"]{ font-variant-numeric:tabular-nums; }
/* Числа в таблицах — табличные (ровные колонки цифр), критично для финансов */
[data-testid="stDataFrame"] [role="gridcell"]{ font-variant-numeric:tabular-nums; }

/* Сайдбар — «рэк» прибора */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#F2F5F9,#EDF0F5); border-right:1px solid var(--mx-border); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { font-size:1rem; }

/* Адаптивность: на узких экранах убираем «липкий» hover-подъём и уплотняем каналы пульта */
@media (max-width: 900px){
  .mixer-channel{ padding:8px 6px; }
  .mixer-channel .ch-label{ font-size:.62rem; }
  .mx-card{ padding:13px 14px; }
  .mx-h .s{ display:none; }               /* моно-подзаголовок прячем, чтобы не переносился */
}
@media (max-width: 640px){
  .mx-card:hover{ transform:none; }        /* на тач-экранах подъём карточки мешает */
  h1{ font-size:1.5rem; } h2{ font-size:1.2rem; }
}
@media (hover: none){                        /* тач-устройства: без hover-трансформаций */
  .mx-card:hover{ transform:none; box-shadow:var(--mx-shadow); }
}

@media (prefers-reduced-motion: reduce){ * { transition:none !important; } .mx-card:hover{ transform:none; } }
</style>
""", unsafe_allow_html=True)

# Те же токены для ВСТРОЕННОЙ тёмной темы Streamlit (подмешиваются apply_theme()).
_DARK_VARS_CSS = """
<style>
:root{
  --mx-ink:#E9EDF3; --mx-ink-2:#C2C9D4; --mx-muted:#949DAB; --mx-faint:#8A93A1;
  --mx-surface:#161B23; --mx-surface-2:#1E242E; --mx-bg:#0C0F14;
  --mx-border:#262D38; --mx-border-2:#333B47;
  --mx-accent:#4D7BFF; --mx-accent-2:#6E93FF; --mx-accent-soft:#1A2236; --mx-accent-ring:rgba(77,123,255,.22);
  --mx-live:#E0A24A; --mx-live-soft:#2A2415;
  --mx-pos:#36D399; --mx-neg:#FF6B81; --mx-warn:#E0A24A;
  --mx-pos-bg:#13271E; --mx-neg-bg:#2A1820; --mx-warn-bg:#2A2415;
  --mx-shadow:0 1px 2px rgba(0,0,0,.4); --mx-shadow-lift:0 12px 34px rgba(0,0,0,.5);
}
.stApp{ background:radial-gradient(1200px 600px at 100% -10%, rgba(77,123,255,.08), transparent 60%), var(--mx-bg); }
.mx-tag-leaf{ background:#16294A; color:#86B0F5; }
.mx-tag-parent{ background:#352713; color:#E0A24A; }
.mx-meter .fill.plan{ background:repeating-linear-gradient(90deg,#39414E,#39414E 6px,#454E5C 6px,#454E5C 12px); }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#111620,#0E121A); }
/* === ИНТЕГРАЦИЯ СО СТАНДАРТНЫМИ ЭЛЕМЕНТАМИ STREAMLIT === */

/* Базовый текст, списки, подписи полей, радио-кнопок и чекбоксов */
.stMarkdown p, .stMarkdown li, label, div[data-testid="stCaptionContainer"] p {
    color: var(--mx-ink) !important;
}

/* Цвет текста заголовков внутри экспандеров */
[data-testid="stExpander"] summary p {
    color: var(--mx-ink) !important;
}

/* Фон и цвет текста внутри полей ввода (input, selectbox) */
[data-baseweb="input"] input, .stSelectbox div[data-baseweb="select"] {
    background-color: var(--mx-surface) !important;
    color: var(--mx-ink) !important;
}

/* Фон и текст выпадающих списков (меню Selectbox) */
ul[data-baseweb="menu"] {
    background-color: var(--mx-surface) !important;
}
ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] span {
    color: var(--mx-ink) !important;
}

/* Текст на вкладках (Tabs) */
.stTabs [data-baseweb="tab"] p {
    color: var(--mx-muted) !important;
}
.stTabs [aria-selected="true"] p {
    color: var(--mx-accent) !important;
}
</style>
"""


def theme_is_dark() -> bool:
    """Определяет активную тему исключительно через нативные инструменты Streamlit.
    Используется только для СЕРВЕРНОГО контента (палитра таблиц, цвета графа), который
    нельзя переключить чистым CSS. Основное оформление темы — клиентское (см. apply_theme)."""
    try:
        # Для современных версий Streamlit (1.37+)
        t = getattr(getattr(st, "context", None), "theme", None)
        if t is not None and getattr(t, "type", None):
            return str(t.type).lower() == "dark"
    except Exception:
        pass
    try:
        # Фолбэк для старых версий
        return str(st.get_option("theme.base") or "light").lower() == "dark"
    except Exception:
        return False


# КЛИЕНТСКАЯ тёмная тема: те же правила, но внутри @media (prefers-color-scheme: dark).
# Переключение темы в меню Streamlit НЕ перезапускает Python-скрипт, поэтому серверная
# вставка «если тема тёмная» не срабатывала до перезагрузки. Media-query решает это в
# браузере МГНОВЕННО и без перезапуска — переменные --mx-* меняются синхронно с темой.
_inner = _DARK_VARS_CSS.replace("<style>", "", 1).rsplit("</style>", 1)[0]
_DARK_VARS_CSS_CLIENT = "<style>@media (prefers-color-scheme: dark){" + _inner + "}</style>"


def apply_theme():
    """Вставляет тёмные стили как @media (prefers-color-scheme: dark) — ВСЕГДА.
    Браузер сам применит их при тёмной теме, без перезапуска скрипта и без перезагрузки."""
    st.markdown(_DARK_VARS_CSS_CLIENT, unsafe_allow_html=True)


# палитра строк таблицы прогноза — зависит от темы (для читаемости в тёмной теме)
def table_palette():
    if theme_is_dark():
        return {'year': 'background-color:#23263A; color:#E6E8EE; font-weight:700;',
                'locked': 'color:#7B8494; background-color:#181B22;',
                'neg': 'background-color:#2A1920; color:#FF8A9B; font-weight:600;',
                'pos': 'background-color:#15271E; color:#5FE0A0; font-weight:600;'}
    return {'year': 'background-color:#EEF0F6; color:#161A22; font-weight:700;',
            'locked': 'color:#9AA1B2; background-color:#F6F7FA;',
            'neg': 'background-color:#FBE9EC; color:#9A1C2E; font-weight:600;',
            'pos': 'background-color:#E7F5EE; color:#0F6B3A; font-weight:600;'}


def _fnum(v, default=0.0):
    """Безопасное приведение значения ячейки к неотрицательному float. Удалённая ячейка
    матрицы приходит как None (или пустая строка) — превращаем в 0.0, не роняя приложение."""
    try:
        f = float(v) if v is not None and v != "" else float(default)
    except (TypeError, ValueError):
        f = float(default)
    return f if f > 0 else 0.0


def fmt(x, nd=2):
    """Форматирование числа в русском стиле: пробел — разделитель тысяч, запятая — десятичная."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    s = f"{v:,.{nd}f}"
    return s.replace(",", "\u00A0").replace(".", ",")


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# Надстрочные символы для степени (в т.ч. минус) — вид «3×10⁻⁵».
_SUP = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def fmt_sci(v, sig=2):
    """Число как «целое × 10^степень» (научная запись): рычаг/эластичность/дельта.
    Мантисса — целое из `sig` значащих цифр, степень — надстрочными (3×10⁻⁵)."""
    import math
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    neg = v < 0
    a = abs(v)
    exp = math.floor(math.log10(a))
    intmant = int(round(a / (10 ** exp) * (10 ** (sig - 1))))  # целое из sig цифр
    exp2 = exp - (sig - 1)
    if intmant >= 10 ** sig:            # перенос при округлении вверх (9.99→10)
        intmant //= 10; exp2 += 1
    while intmant % 10 == 0 and intmant != 0:  # убираем хвостовые нули мантиссы
        intmant //= 10; exp2 += 1
    sign = "\u2212" if neg else ""      # настоящий минус «−»
    if exp2 == 0:                       # без ×10⁰ для «круглых» значений
        return f"{sign}{intmant}"
    return f"{sign}{intmant}\u00d710{str(exp2).translate(_SUP)}"


def table_download(df, filename: str, key: str, label: str = "⬇ Скачать таблицу (CSV)"):
    """Явная кнопка скачивания CSV под таблицей (в дополнение к встроенному тулбару
    разворота/скачивания, который появляется при наведении на правый верх таблицы)."""
    try:
        import pandas as _pd
        if not isinstance(df, _pd.DataFrame):
            return
        csv = df.to_csv(index=False).encode('utf-8-sig')  # utf-8-sig — корректная кириллица в Excel
        st.download_button(label, data=csv, file_name=filename, mime="text/csv",
                           key=key, use_container_width=True)
    except Exception:
        pass


def pct_badge(p):
    cls = "flat"
    if p > CHANGE_THRESHOLD:
        cls = "pos"
    elif p < -CHANGE_THRESHOLD:
        cls = "neg"
    sign = "+" if p >= 0 else ""
    return f'<span class="mx-badge {cls}">{sign}{p*100:.2f}%</span>'


def section(title, sub=""):
    """Заголовок секции с акцентной чертой."""
    s = f'<span class="s">· {sub}</span>' if sub else ""
    st.markdown(f'<div class="mx-h"><span class="t">{title}</span>{s}</div>', unsafe_allow_html=True)


def chip(text, kind="info"):
    return f'<span class="mx-chip {kind}"><span class="dot"></span>{text}</span>'


def _vis_network_tags() -> str:
    """CSS+JS для vis-network. Если локальные файлы assets/vis-network.min.{css,js} есть —
    ВСТРАИВАЕМ их прямо в HTML (работает офлайн, без интернета). Иначе — ссылки на CDN."""
    css = _read_asset("assets", "vis-network.min.css")
    js = _read_asset("assets", "vis-network.min.js")
    if css and js:
        return f"<style>{css}</style><script>{js}</script>"
    # Фолбэк для онлайн-разработки (на офлайн-ПК не сработает — положите файлы в assets/)
    return ('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/dist/vis-network.min.css"/>'
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>')


def build_goal_graph_html(engine, height_px: int = 560, dark: bool = False) -> str:
    """Интерактивный граф дерева целей через vis-network (CDN, без установки пакетов).
    Узлы: KPI (прямоугольники) и работы. КРАСНЫЕ — «проседающие» (работа завершается позже плана
    или у KPI прогноз ниже плана). Толщина рёбер — вес связи (влияние). Рендерится в браузере."""
    G = engine.G
    nodes_js, edges_js = [], []
    edge_w = {}
    for kpi_id, wd in getattr(engine, 'kpi_weights', {}).items():
        for (s, t), d in wd.items():
            w = float(d.get('weight', 0.0) or 0.0)
            edge_w[(str(s), str(t))] = max(edge_w.get((str(s), str(t)), 0.0), w)

    def _late(nid, a):
        try:
            pe = a.get('T_plan_end') or a.get('T_end')
            return engine._pdate(a.get('T_end')) > engine._pdate(pe)
        except Exception:
            return False

    import networkx as _nx
    def _kpi_has_late(kpi):
        # KPI «проседает», если хотя бы один питающий лист завершается позже плана
        try:
            anc = _nx.ancestors(G, kpi)  # все узлы, ведущие к KPI (рёбра дети→родитель→KPI)
        except Exception:
            anc = set()
        for d in anc:
            ad = G.nodes[d]
            if G.in_degree(d) == 0 and _late(d, ad):
                return True
        return False

    nb = '#2A323E' if dark else '#0b1220'        # рамка узлов
    nfont = '#E9EDF3' if dark else '#1b2230'     # цвет подписи работ
    parent_c = '#5A6678' if dark else '#8DA2C0'  # составной узел
    edge_c = '#3a4452' if dark else '#9fb0c8'    # рёбра
    bg = '#11151c' if dark else '#ffffff'        # фон контейнера
    bd = '#262D38' if dark else '#d8deea'        # рамка контейнера
    for n, a in G.nodes(data=True):
        is_kpi = str(a.get('type', '')).upper() == 'KPI'
        raw = str(a.get('name', n)).strip()
        nm = raw if len(raw) <= 22 else raw[:21] + '…'   # короткое имя, чтобы не налезало
        if is_kpi:
            color = '#D8425A' if _kpi_has_late(n) else '#0E8F5E'
            nodes_js.append({'id': str(n), 'label': nm, 'shape': 'box', 'margin': 10,
                             'widthConstraint': {'maximum': 150},
                             'color': {'background': color, 'border': nb},
                             'font': {'color': '#fff', 'face': 'Inter', 'size': 13, 'multi': False}})
        else:
            under = _late(n, a)
            leaf = G.in_degree(n) == 0
            color = '#D8425A' if under else ('#1B4DFF' if leaf else parent_c)
            nodes_js.append({'id': str(n), 'label': f"{n}\n{nm}", 'shape': 'dot',  # настоящий перенос строки
                             'size': 11 + (5 if leaf else 0),
                             'widthConstraint': {'maximum': 140},
                             'color': {'background': color, 'border': nb},
                             # подложка под подписью — чтобы текст соседних узлов не сливался
                             'font': {'face': 'Inter', 'size': 12, 'color': nfont,
                                      'background': bg, 'strokeWidth': 0, 'vadjust': 2, 'multi': False}})
    for s, t in G.edges():
        w = edge_w.get((str(s), str(t)), 0.0)
        edges_js.append({'from': str(s), 'to': str(t), 'arrows': 'to',
                         'width': round(1.0 + 5.0 * w, 2),
                         'color': {'color': edge_c, 'opacity': 0.8},
                         'smooth': {'type': 'cubicBezier'}})
    payload = json.dumps({'nodes': nodes_js, 'edges': edges_js}, ensure_ascii=False)
    # АВТО-ПОДБОР интервалов: горизонтальный — от самой длинной подписи (чтобы не налезали),
    # вертикальный — больше для двухстрочных подписей работ; вид подгоняется под контейнер.
    n_total = len(nodes_js)
    has_work = any('\n' in (nd.get('label') or '') for nd in nodes_js)
    longest = max((max((len(line) for line in str(nd.get('label', '')).split('\n')), default=0)
                   for nd in nodes_js), default=8)
    font_sz = 12 if n_total <= 80 else (11 if n_total <= 200 else 10)
    node_spacing = int(min(300, max(120, longest * 8.5 + 30)))       # ширина под подпись
    level_sep = 135 + (30 if has_work else 0) + (15 if longest > 16 else 0)
    tree_spacing = node_spacing + 40
    return (
        '<div id="goalnet" style="height:%dpx;border:1px solid %s;border-radius:12px;background:%s"></div>'
        '%s'  # vis-network: локально (офлайн) или CDN
        '<script>(function(){var data=%s;function draw(){if(typeof vis==="undefined"){setTimeout(draw,200);return;}'
        'var c=document.getElementById("goalnet");'
        'data.nodes.forEach(function(x){if(x.font)x.font.size=%d;});'
        'var net=new vis.Network(c,{nodes:new vis.DataSet(data.nodes),edges:new vis.DataSet(data.edges)},'
        '{layout:{hierarchical:{enabled:true,direction:"DU",sortMethod:"directed",'
        'levelSeparation:%d,nodeSpacing:%d,treeSpacing:%d,blockShifting:true,edgeMinimization:true}},'
        'physics:{enabled:false},interaction:{hover:true,tooltipDelay:120,navigationButtons:true,keyboard:false}});'
        'net.once("afterDrawing",function(){net.fit({animation:false});});'
        '}draw();})();</script>'
    ) % (height_px, bd, bg, _vis_network_tags(), payload, font_sz, level_sep, node_spacing, tree_spacing)


def meter_html(plan, forecast):
    """Фейдер-метр пульта: общая шкала, уровень прогноза с «головкой» и засечка плана."""
    plan = max(0.0, safe_float(plan)); forecast = max(0.0, safe_float(forecast))
    mx = max(plan, forecast, 1e-9)
    pw = plan / mx * 100.0
    fw = forecast / mx * 100.0
    cls = "acc"
    if forecast > plan + 0.01: cls = "pos"
    elif forecast < plan - 0.01: cls = "neg"
    return (
        '<div class="mx-meter">'
        f'<div class="cap"><span>план</span><span>{fmt(plan)}</span></div>'
        f'<div class="bar"><div class="fill plan" style="width:{pw:.1f}%"></div></div>'
        f'<div class="cap"><span>прогноз</span><span>{fmt(forecast)}</span></div>'
        f'<div class="bar"><div class="fill {cls}" style="width:{fw:.1f}%"></div>'
        f'<div class="tick" style="left:{pw:.1f}%"></div></div>'
        '</div>'
    )


def fmt_date(s):
    """Дата 'YYYY-MM-DD' → 'дд.мм.гггг'."""
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").strftime("%d.%m.%Y")
    except Exception:
        return str(s)


# ======================================================================
# ЗАГОЛОВОК
# ======================================================================

# ======================================================================
# ИНИЦИАЛИЗАЦИЯ ЯДРА (кеш на проект + подпись файлов)
# ======================================================================
import threading
import time as _time

def _build_engine(slug, flags, cfg, progress_cb):
    """Тяжёлая сборка движка проекта (методики + веса). Вызывается в ФОНОВОМ потоке.

    Логи сборки идут в файл этого проекта (set_log_project в начале потока).
    progress_cb(этап, detail, frac) обновляет состояние для страницы загрузки."""
    ctx = ps.context_for(slug)
    mc.set_log_project(slug, ctx.log_path)  # лог этого проекта (и в фоне тоже)
    if not ctx.has_schedule():
        raise RuntimeError(f"Не найден план-график проекта: {ctx.schedule_path}")
    ctx.ensure_dirs()
    progress_cb("Загрузка план-графика и показателей", frac=0.03)
    df_nodes, df_edges, kpi_meths, unmatched_fin = DataLoaderOrchestrator.build_system_context(
        ctx.schedule_path, ctx.indicators_path, ctx.methodologies_dir, 
        ctx.finances_path if ctx.has_finances() else None
    )
    llm_cfg = ps.load_llm_settings()
    llm = LocalLLMEngine(
        base_url=llm_cfg.get("base_url") or "http://localhost:11434/v1",
        api_key=llm_cfg.get("api_key") or "local",
        model_name=llm_cfg.get("model") or "gpt-oss:20b",
        timeout=llm_cfg.get("timeout") or 120,
        enabled=bool(llm_cfg.get("enabled", True)),
        use_meth_cache=flags['use_m_cache'], meth_cache_dir=ctx.meth_cache_dir,
        force_recompress=flags['force'],
    )
    engine = ProjectMixer(
        df_nodes, df_edges, llm_engine=llm, methodologies=kpi_meths,
        unmatched_finances=unmatched_fin, # <--- ПЕРЕДАЕМ В ЯДРО
        config=cfg, use_cached_weights=flags['use_w_cache'], weights_path=ctx.weights_path,
        progress_callback=progress_cb,
    )
    engine.apply_state(ps.load_baseline(ctx))  # утверждённый план поверх Excel
    # Ручные флаги «Финансовая веха» (переживают пересборку): применяем и пересчитываем
    # распределение денег, если что-то реально поменялось.
    try:
        _ffl = ps.load_project_settings(ctx).get('financial_flags', {}) or {}
        _chg = False
        for _nid, _fl in _ffl.items():
            if _nid in engine.G.nodes and bool(engine.G.nodes[_nid].get('is_financial', False)) != bool(_fl):
                engine.G.nodes[_nid]['is_financial'] = bool(_fl)
                _chg = True
        if _chg:
            engine.recalculate()
    except Exception:
        pass
    return engine, llm


class ProjectBuildManager:
    """Менеджер фоновой сборки проектов.

    Пока новый проект собирается в отдельном потоке (видно на странице загрузки),
    уже собранные проекты остаются доступны для работы. Состояние каждого проекта
    защищено блокировкой; готовый движок хранится в менеджере на время жизни процесса."""
    def __init__(self):
        self._lock = threading.Lock()
        self._states = {}

    def get(self, slug):
        with self._lock:
            s = self._states.get(slug)
            return dict(s) if s else None

    def request(self, slug, signature, builder):
        """Запускает сборку, если её ещё нет или изменилась подпись (структура/настройки)."""
        with self._lock:
            s = self._states.get(slug)
            if s and s.get('signature') == signature and s.get('status') in ('running', 'ready'):
                return s['status']
            self._states[slug] = {'status': 'running', 'signature': signature, 'frac': 0.0,
                                  'stage': 'В очереди…', 'detail': '', 'engine': None, 'llm': None,
                                  'error': None, 'started': _time.time(), 'finished': None}
        if os.environ.get("MIXER_SYNC_BUILD") == "1":
            self._run(slug, signature, builder)  # синхронно (тесты/отладка)
        else:
            threading.Thread(target=self._run, args=(slug, signature, builder), daemon=True).start()
        cur = self.get(slug)
        return cur.get('status') if cur else 'running'

    def _run(self, slug, signature, builder):
        def cb(stage, detail='', frac=None):
            with self._lock:
                s = self._states.get(slug)
                if not s or s.get('signature') != signature:
                    return
                s['stage'] = stage
                if detail:
                    s['detail'] = detail
                if frac is not None:
                    s['frac'] = max(0.0, min(1.0, float(frac)))
        try:
            engine, llm = builder(cb)
            with self._lock:
                s = self._states.get(slug)
                if s and s.get('signature') == signature:
                    s.update(status='ready', engine=engine, llm=llm, frac=1.0,
                             stage='Готово', finished=_time.time())
        except BaseException as e:
            import traceback
            err = f"{e}"
            try:
                mc.logger.error(f"Сборка проекта {slug} не удалась: {e}\n{traceback.format_exc()}")
            except Exception:
                pass
            with self._lock:
                s = self._states.get(slug)
                if s and s.get('signature') == signature:
                    s.update(status='error', error=err, stage='Ошибка', finished=_time.time())

    def adopt(self, slug, signature):
        """Принять текущий движок как соответствующий новой подписи (после commit —
        движок уже изменён в памяти, пересобирать не нужно)."""
        with self._lock:
            s = self._states.get(slug)
            if s and s.get('status') == 'ready':
                s['signature'] = signature

    def invalidate(self, slug):
        with self._lock:
            self._states.pop(slug, None)


@st.cache_resource
def get_build_manager() -> ProjectBuildManager:
    return ProjectBuildManager()


def _build_signature(slug, ctx, cfg, flags):
    """Подпись сборки БЕЗ учёта baseline: commit (запись baseline) не должен вызывать
    пересборку — движок уже изменён в памяти. Меняют подпись только файлы план-графика/
    показателей/методик/финансов, коэффициенты и флаги кеша."""
    
    # ИСПРАВЛЕНО: Распаковываем 5 значений, включая fin (финансы)
    sched, ind, fin, meth, _base = ctx.file_signature()
    
    _llm = ps.load_llm_settings()
    # Смена провайдера/модели/включённости влияет на ИИ-результаты → пересборка;
    # смена токена/таймаута на структуру результата не влияет (в подпись не входят).
    llm_sig = (_llm.get('enabled'), _llm.get('provider'), _llm.get('base_url'), _llm.get('model'))
    
    # ИСПРАВЛЕНО: Добавляем fin в итоговый кортеж подписи
    return (slug, sched, ind, fin, meth, llm_sig,
            cfg.alpha, cfg.beta, cfg.lambda_f, cfg.budget_scale, cfg.sigmoid_k, cfg.activation_threshold,
            cfg.sigmoid_activation_k, cfg.deficit_penalty_power, cfg.time_penalty_power,
            cfg.time_bonus_enabled,
            cfg.late_finish_penalty_enabled, cfg.late_finish_weight, cfg.default_agg_mode,
            cfg.default_ces_rho,
            cfg.forecast_mode,
            # Дисконт и базовый год приведения — реальные параметры математики: их читает
            # _evaluate_node_finances (f_real) и через него вся ценность/прогноз KPI. Без них
            # в подписи сдвиг ползунков «Ставка дисконтирования»/«Базовый год» в боковой панели
            # не пересобирал движок и не влиял на расчёт (менялся лишь предпросмотр во фрагменте).
            cfg.discount_rate, cfg.base_year,
            flags['use_w_cache'], flags['use_m_cache'], flags['force'])


def _llm_settings_panel():
    """Глобальная настройка подключения к ИИ: локальная Ollama или любой OpenAI-совместимый
    API (OpenAI, Together, Groq, OpenRouter, LM Studio…). Поля: провайдер, base_url, токен,
    модель, таймаут, переключатель «использовать ИИ». Проверка подключения и сохранение."""
    s = ps.load_llm_settings()
    ss = st.session_state
    for k, v in {'_llm_enabled': s['enabled'], '_llm_provider': s['provider'],
                 '_llm_base': s['base_url'], '_llm_key': s['api_key'],
                 '_llm_model': s['model'], '_llm_timeout': int(s.get('timeout', 120))}.items():
        ss.setdefault(k, v)

    cur = s.get('provider', 'local')
    badge = "включён" if s['enabled'] else "выключен"
    prov_label = ps.LLM_PRESETS.get(cur, {}).get('label', cur)
    with st.expander(f"⚙️ Подключение к ИИ — {prov_label} · {badge}", expanded=False):
        st.toggle("Использовать ИИ", key='_llm_enabled',
                  help="Выключите, если работаете только по математической модели и кешам — "
                       "тогда обращений к модели не будет.")
        prov_keys = list(ps.LLM_PRESETS.keys())
        st.selectbox("Провайдер", prov_keys, key='_llm_provider',
                     format_func=lambda k: ps.LLM_PRESETS[k]['label'])
        if st.button("↻ Подставить адрес и модель из пресета", key='_llm_preset_fill'):
            pr = ps.LLM_PRESETS[ss['_llm_provider']]
            ss['_llm_base'] = pr['base_url']
            ss['_llm_model'] = pr['model']
            st.rerun()
        st.text_input("Адрес API (base_url)", key='_llm_base',
                      placeholder="http://localhost:11434/v1  или  https://api.openai.com/v1",
                      help="Должен оканчиваться на /v1 для OpenAI-совместимых API.")
        st.text_input("Токен / API-ключ", key='_llm_key', type="password",
                      help="Для локальной Ollama не нужен — оставьте пустым.")
        st.text_input("Модель", key='_llm_model',
                      placeholder="gpt-oss:20b  ·  gpt-4o-mini  ·  llama-3.1-70b …")
        st.number_input("Таймаут запроса, сек", min_value=5, max_value=900,
                        step=5, key='_llm_timeout')

        c1, c2 = st.columns(2)
        if c1.button("Проверить подключение", key='_llm_test', use_container_width=True):
            import tempfile
            try:
                eng = mc.LocalLLMEngine(
                    base_url=ss['_llm_base'] or "http://localhost:11434/v1",
                    api_key=ss['_llm_key'] or "local", model_name=ss['_llm_model'] or "gpt-oss:20b",
                    timeout=int(ss['_llm_timeout']), enabled=bool(ss['_llm_enabled']),
                    use_meth_cache=False, meth_cache_dir=tempfile.mkdtemp())
                ok, msg = eng.test_connection()
            except Exception as e:
                ok, msg = False, f"Ошибка инициализации клиента: {str(e)[:200]}"
            (st.success if ok else st.error)(msg)
        if c2.button("Сохранить настройки", type="primary", key='_llm_save', use_container_width=True):
            ps.save_llm_settings({
                'enabled': bool(ss['_llm_enabled']), 'provider': ss['_llm_provider'],
                'base_url': ss['_llm_base'], 'api_key': ss['_llm_key'],
                'model': ss['_llm_model'], 'timeout': int(ss['_llm_timeout'])})
            mgr = get_build_manager()
            for c in ps.discover(ps.PROJECTS_ROOT):
                mgr.invalidate(c.slug)  # ИИ сменился → пересобрать при открытии
            st.success("Сохранено. Открытые проекты пересоберутся с новым подключением.")
            st.rerun()
        st.caption("Токен хранится локально в открытом виде (`projects/llm_settings.json`). "
                   "Подходит любой OpenAI-совместимый сервис: OpenAI, Together, Groq, OpenRouter, LM Studio.")


def render_loading_page(slug, ctx, mgr):
    """Страница загрузки проекта с прогрессом. Пока проект собирается, можно вернуться
    к списку и работать с уже загруженными проектами."""
    apply_theme()
    state = mgr.get(slug) or {}
    st.markdown(f"# ⏳ Готовится проект: {ctx.title}")
    st.markdown('<div class="mx-sub">Сжимаются методики и рассчитываются веса влияния. '
                'Это разовая операция — результат кешируется.</div>', unsafe_allow_html=True)
    frac = float(state.get('frac', 0.0) or 0.0)
    try:
        st.progress(min(1.0, max(0.0, frac)), text=f"{state.get('stage', 'Подготовка…')} — {int(frac*100)}%")
    except Exception:
        st.progress(min(1.0, max(0.0, frac)))
    if state.get('detail'):
        st.caption(state['detail'])
    qd = mc.llm_queue_depth()
    if qd > 1:
        st.caption(f"⏳ Запросов к ИИ в очереди: {qd} (модель обрабатывает по одному — без перегрузки).")
    st.info("Можно вернуться к списку проектов и продолжить работу с уже загруженными — "
            "этот соберётся в фоне.")
    if st.button("← К списку проектов", use_container_width=True):
        st.session_state.active_project = None
        st.rerun()
    # авто-обновление прогресса
    _time.sleep(0.6)
    st.rerun()


# ======================================================================
# СТРАНИЦА «ПРОЕКТЫ»
# ======================================================================
def page_projects():
    import datetime as _dt
    apply_theme()  # #9: тёмная тема
    st.markdown("# 📁 Проекты")
    st.markdown('<div class="mx-sub">Каждый проект — свой план-график, показатели и методики. '
                'Откройте существующий или создайте новый.</div>', unsafe_allow_html=True)
    _llm_settings_panel()

    projects = ps.discover(ps.PROJECTS_ROOT)
    if not projects:
        st.info("Пока нет ни одного проекта. Создайте новый ниже и загрузите файлы.")
    else:
        _mgr = get_build_manager()
        _busy = any((_mgr.get(c.slug) or {}).get('status') == 'running' for c in projects)
        cols = st.columns(3)
        for i, c in enumerate(projects):
            with cols[i % 3]:
                lm = c.last_modified()
                when = _dt.datetime.fromtimestamp(lm).strftime('%d.%m.%Y') if lm else '—'
                files = ("график" if c.has_schedule() else "нет графика")
                files += " · показатели" if c.has_indicators() else " · нет показателей"
                if c.has_finances(): files += " · финансы"
                nmeth = len(c.methodology_files())
                base = "✓ утверждён план" if c.has_baseline() else "план из Excel"
                bst = _mgr.get(c.slug) or {}
                status = bst.get('status')
                if status == 'running':
                    chip_html = f'<span class="mx-chip info"><span class="dot"></span>готовится {int(float(bst.get("frac", 0) or 0)*100)}%</span>'
                elif status == 'ready':
                    chip_html = '<span class="mx-chip ok"><span class="dot"></span>загружен</span>'
                elif status == 'error':
                    chip_html = '<span class="mx-chip bad"><span class="dot"></span>ошибка сборки</span>'
                else:
                    chip_html = '<span class="mx-chip"><span class="dot" style="background:var(--mx-faint)"></span>не загружен</span>'
                base_chip = ('<span class="mx-pill mx-tag-leaf">план утверждён</span>' if c.has_baseline()
                             else '<span class="mx-pill">план из Excel</span>')
                st.markdown(
                    f'<div class="mx-card" style="margin-bottom:8px">'
                    f'<div class="lbl" style="margin-bottom:8px">проект</div>'
                    f'<div class="name" style="font-family:var(--mx-display); font-size:1.12rem; min-height:2.5em; margin-bottom:10px">{c.title}</div>'
                    f'<div class="mx-ribbon" style="margin-bottom:8px">{chip_html}{base_chip}</div>'
                    f'<div class="plan">{files} · методик: {nmeth}</div>'
                    f'<div class="plan">изменён: {when}</div></div>',
                    unsafe_allow_html=True,
                )
                open_lbl = "Открыть" if status != 'running' else "Открыть (идёт сборка)"
                if st.button(open_lbl, key=f"open_{c.slug}", type="primary", use_container_width=True):
                    st.session_state.active_project = c.slug
                    st.rerun()
                if st.button("Удалить", key=f"del_{c.slug}", use_container_width=True):
                    _mgr.invalidate(c.slug)
                    mc.unregister_project_log(c.slug)  # закрыть логовый хендлер проекта
                    ps.delete_project(c)
                    if st.session_state.get('active_project') == c.slug:
                        st.session_state.active_project = None
                    st.rerun()
        if _busy:
            st.caption("Идёт фоновая сборка одного из проектов — готовые проекты доступны, "
                       "статус обновится автоматически.")
            _time.sleep(1.0)
            st.rerun()

    st.divider()
    section("Новый проект", "создать проект и загрузить файлы")
    with st.container(border=True):
        new_title = st.text_input("Название проекта", key="new_proj_title",
                                  placeholder="например: Цифровая трансформация")
        up_s = st.file_uploader("План-график (.xlsx)", type=["xlsx"], key="up_sched")
        up_f = st.file_uploader("Фин.обеспечение (.xlsx)", type=["xlsx"], key="up_fin")
        up_i = st.file_uploader("Плановые показатели (.xlsx)", type=["xlsx"], key="up_ind")
        up_m = st.file_uploader("Методики (.docx/.pdf/.txt, можно несколько)",
                                type=["docx", "pdf", "txt", "doc"],
                                accept_multiple_files=True, key="up_meth")
        if st.button("Создать проект", type="primary"):
            if not (new_title or "").strip():
                st.warning("Укажите название проекта.")
            elif up_s is None:
                st.warning("Нужен хотя бы файл план-графика.")
            else:
                c = ps.create_project(new_title.strip())
                ps.save_schedule(c, up_s.getvalue())
                if up_f is not None:
                    ps.save_finances(c, up_f.getvalue())
                if up_i is not None:
                    ps.save_indicators(c, up_i.getvalue())
                for mf in (up_m or []):
                    ps.save_methodology(c, mf.name, mf.getvalue())
                st.session_state.active_project = c.slug
                st.success(f"Проект «{c.title}» создан.")
                st.rerun()

    if projects:
        section("Импорт файлов в существующий проект")
        with st.container(border=True):
            tgt_title = st.selectbox("Проект", [c.title for c in projects], key="imp_tgt")
            tctx = next((c for c in projects if c.title == tgt_title), None)
            
            i_s = st.file_uploader("План-график (.xlsx)", type=["xlsx"], key="imp_sched")
            i_i = st.file_uploader("Показатели (.xlsx)", type=["xlsx"], key="imp_ind")
            i_m = st.file_uploader("Методики", type=["docx", "pdf", "txt", "doc"],
                                   accept_multiple_files=True, key="imp_meth")
            i_f = st.file_uploader("Фин.обеспечение (.xlsx)", type=["xlsx"], key="imp_fin")
            if st.button("Загрузить в проект") and tctx is not None:
                if i_s is not None:
                    ps.save_schedule(tctx, i_s.getvalue())
                if i_i is not None:
                    ps.save_indicators(tctx, i_i.getvalue())
                if i_f is not None:
                    ps.save_finances(tctx, i_f.getvalue())
                for mf in (i_m or []):
                    ps.save_methodology(tctx, mf.name, mf.getvalue())
                get_build_manager().invalidate(tctx.slug)
                st.success("Файлы загружены. Откройте проект, чтобы пересобрать модель.")
                st.rerun()


# ======================================================================
# СТРАНИЦА «МИКШЕР» (активный проект)
# ======================================================================

def _apply_transfer_deltas(selected_entity, active_years, state_key):
    """Инкрементально переносит деньги между 'Потребность'/'Доп. потребность' и 'База' по
    ползункам Каналов 7/8 (% от исходной суммы года). ОБЩАЯ функция для пульта (фрагмент,
    который перерисовывается САМ ПО СЕБЕ при перетаскивании ползунка) и для основного тела
    страницы: если считать перенос в двух местах отдельно, копия внутри фрагмента не увеличивала
    table_nonce — а без этого таблица (data_editor с ключом на table_nonce) не перечитывает
    новые данные при следующей полной перерисовке, и «Применить сценарий» уходит по старым,
    ещё не перенесённым цифрам."""
    if not active_years:
        return False
    changed = False
    for y_str in active_years:
        kr = f"ch_trans_req_{selected_entity}_{y_str}"
        ka = f"ch_trans_add_{selected_entity}_{y_str}"
        l_kr = f"_last_val_{kr}"
        l_ka = f"_last_val_{ka}"

        curr_req = st.session_state.get(kr, 0)
        curr_add = st.session_state.get(ka, 0)
        last_req = st.session_state.get(l_kr, 0)
        last_add = st.session_state.get(l_ka, 0)

        if curr_req != last_req or curr_add != last_add:
            changed = True
            delta_req = (curr_req - last_req) / 100.0
            delta_add = (curr_add - last_add) / 100.0

            st.session_state[l_kr] = curr_req
            st.session_state[l_ka] = curr_add

            for r in st.session_state[state_key]:
                if r["Год"] == y_str:
                    amt_req = _fnum(r.get("_orig_req")) * delta_req
                    amt_add = _fnum(r.get("_orig_add")) * delta_add
                    r["База"] += (amt_req + amt_add)
                    r["Потребность"] -= amt_req
                    r["Доп. потребность"] -= amt_add

            st.session_state.table_nonce += 1
    return changed


# === ГЛОБАЛЬНЫЙ ФРАГМЕНТ ПУЛЬТА M-Vave (изолированный, живёт на своём таймере) ===
# УЛУЧШЕНИЕ: автообновление каждые 0.15с включается только при наличии MIDI-пакетов —
# без пульта нет смысла крутить rerun-цикл и жечь CPU.
@st.fragment(run_every="0.25s" if MIDI_AVAILABLE else None)
def realtime_console_fragment(selected_entity, node, engine, def_rho_req, def_rho_add, cur_start, cur_end, is_ms, active_years, budget_unit, infl, auto_balance, disc_rate=0.06, disc_base_year=2026):
    import time
    from plotly.subplots import make_subplots

    # Данные конкретной задачи — из отдельного ключа
    _entity_state_key = f"finance_profile_{selected_entity}"
    if _entity_state_key not in st.session_state:
        return
    _entity_data = st.session_state[_entity_state_key]

    # --- 1. MIDI SYNC (REAL-TIME) ---
    # Синхронизация состояния MIDI-пульта с сессией Streamlit.
    # Это позволяет пульту управлять слайдерами на экране в реальном времени.
    if MIDI_AVAILABLE:
        midi_state = get_midi_state()
        revision_key = f"_midi_revision_{selected_entity}"
        if midi_state.get('revision') != st.session_state.get(revision_key):
            st.session_state[f"ch_base_{selected_entity}"] = midi_state['base']
            st.session_state[f"ch_req_{selected_entity}"] = midi_state['req']
            st.session_state[f"ch_add_{selected_entity}"] = midi_state['add']
            st.session_state[f"ch_shift_{selected_entity}"] = midi_state['shift']
            
            focus_yr = st.session_state.get(f"midi_focus_year_{selected_entity}")
            if focus_yr:
                st.session_state[f"ch_trans_req_{selected_entity}_{focus_yr}"] = midi_state['trans_req']
                st.session_state[f"ch_trans_add_{selected_entity}_{focus_yr}"] = midi_state['trans_add']
            st.session_state[revision_key] = midi_state['revision']

    # --- 2. CHANNELS 1-4 (MAIN CONSOLE) ---
    st.markdown('<div class="mx-h">🎛️ M-Vave Console (Основные каналы 1-4)</div>', unsafe_allow_html=True)
    channels = st.columns(4) # <--- Уменьшили до 4 колонок

    def draw_channel(col, label, val_key, min_v, max_v, def_v, step, fmt_str, unit):
        with col:
            st.markdown(f'<div class="mixer-channel"><div class="ch-label">{label}</div>', unsafe_allow_html=True)
            unique_key = f"{val_key}_{selected_entity}"
            if unique_key not in st.session_state:
                st.session_state[unique_key] = max(min_v, min(max_v, def_v))
            val = st.slider(label, min_value=min_v, max_value=max_v, step=step, key=unique_key, label_visibility="collapsed")
            if MIDI_AVAILABLE:
                midi_key = val_key.replace("ch_", "")
                if midi_key in get_midi_state() and get_midi_state()[midi_key] != val:
                    get_midi_state()[midi_key] = val
            st.markdown(f'<div class="ch-val">{fmt_str.format(val)}{unit}</div></div>', unsafe_allow_html=True)
            return val

    base_pct = draw_channel(channels[0], "1: БАЗА", "ch_base", 0, 100, 100, 1, "{}", "%") / 100.0
    rho_req  = draw_channel(channels[1], "2: ПОТРЕБ", "ch_req", 0, 100, def_rho_req, 1, "{}", "%") / 100.0
    rho_add  = draw_channel(channels[2], "3: ДОП", "ch_add", 0, 100, def_rho_add, 1, "{}", "%") / 100.0
    shift_m  = draw_channel(channels[3], "4: СДВИГ", "ch_shift", -12, 24, 0, 1, "{:+d}", "м")

    st.divider()

    # --- 3. CHANNELS 5-6 & DYNAMIC GRAPH ---
    st.markdown('<div class="mx-h">🔄 Перераспределение и Живой прогноз</div>', unsafe_allow_html=True)

    cfg = st.session_state.get('custom_config')
    alpha = float(getattr(cfg, 'alpha', 1.0))
    lam = float(getattr(cfg, 'lambda_f', 0.005))
    scale = getattr(cfg, 'budget_scale', 'millions')
    factor = mc.BUDGET_UNITS.get(scale, {'factor': 1.0})['factor']
    lam_eff = lam * factor

    trans_changed = _apply_transfer_deltas(selected_entity, active_years, _entity_state_key)

    # === 1. СБОРКА ФИНАНСОВ И ДАННЫХ ДЛЯ ГРАФИКОВ ===
    updated_fin_dict = {}
    new_F_nominal = 0.0
    
    # Списки для отрисовки столбчатой диаграммы
    plot_years, plot_base, plot_req, plot_add, plot_feff = [], [], [], [], []
    total_feff = 0.0

    for r in _entity_data:
        # ЭФФЕКТИВНЫЕ значения (с учетом ползунка "БАЗА") для передачи в ядро
        b_v_eff = _fnum(r.get('База')) * base_pct
        r_v = _fnum(r.get('Потребность'))
        a_v = _fnum(r.get('Доп. потребность'))
        y_str = str(r['Год'])
        
        updated_fin_dict[y_str] = {'base': b_v_eff, 'req_extra': r_v, 'add': a_v}
        new_F_nominal += b_v_eff + (r_v * rho_req) + (a_v * rho_add)
        
        # Дисконтирование
        df_year = 1.0 / ((1.0 + disc_rate) ** max(0, int(y_str) - disc_base_year))
        
        # ПОЛНЫЕ значения для отрисовки столбцов (весь номинальный бюджет, но дисконтированный)
        b_v_full = _fnum(r.get('База'))
        db_full = b_v_full * df_year
        dr_full = r_v * df_year
        da_full = a_v * df_year
        
        # ЭФФЕКТИВНЫЕ значения для отрисовки линии F_eff (с учётом вероятностей и ползунка БАЗА)
        db_eff = b_v_eff * df_year
        dr_eff = (r_v * rho_req) * df_year
        da_eff = (a_v * rho_add) * df_year
        
        plot_years.append(y_str)
        plot_base.append(db_full)     # Столбцы теперь показывают полный доступный бюджет
        plot_req.append(dr_full)
        plot_add.append(da_full)
        plot_feff.append(db_eff + dr_eff + da_eff) # Линия показывает реально используемый
        total_feff += (db_eff + dr_eff + da_eff)

    # === ВЫЧИСЛЕНИЕ НОВЫХ СРОКОВ (С УЧЕТОМ АВТОБАЛАНСА) ===
    b_dur = max(1, (cur_end - cur_start).days)
    new_start = cur_start + timedelta(days=shift_m * 30)
    slider_dur = float(b_dur)

    # Восстанавливаем настоящий старый бюджет (включая агрегацию для родителей)
    old_fin_dict = {}
    for r in _entity_data:
        y_str = str(r['Год'])
        old_fin_dict[y_str] = {'base': _fnum(r.get('_orig_base')), 'req_extra': _fnum(r.get('_orig_req')), 'add': _fnum(r.get('_orig_add'))}

    _, F_new_real, _ = engine._evaluate_node_finances(updated_fin_dict, rho_req=rho_req, rho_add=rho_add)
    _, F_old_real, _ = engine._evaluate_node_finances(old_fin_dict, rho_req=def_rho_req/100.0, rho_add=def_rho_add/100.0)

    if auto_balance and F_old_real > 0 and F_new_real < F_old_real:
        _cfg = st.session_state.get('custom_config')
        penalty = float(getattr(_cfg, 'time_penalty_power', 1.0))
        safe_f = max(F_new_real, F_old_real * 0.05) 
        stretch_factor = (F_old_real / safe_f) ** penalty
        final_dur = slider_dur * stretch_factor
    else:
        final_dur = slider_dur

    new_end = new_start + timedelta(days=max(1, int(final_dur)))
    if is_ms: 
        new_start = new_end
        
    # === 3. ЖИВОЙ ПРОГНОЗ ЧЕРЕЗ ЯДРО (ДЕНЬГИ + СРОКИ) ===
    sim_res = engine.mix(
        selected_entity, new_F_nominal, 
        new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d"), 
        new_finances=updated_fin_dict, rho_req=rho_req, rho_add=rho_add
    )
    

    # === 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ОТРИСОВКИ ===
    _live_sim = {}
    if infl:
        for k_info in infl:
            k_name = k_info['KPI']
            k_share = k_info['Влияние']
            k_title_wrapped = "<br>".join(textwrap.wrap(k_name, width=38))
            
            # Находим ID показателя
            k_id = next((nid for nid in engine.kpi_ids if nid == k_name or engine.G.nodes[nid].get('name') == k_name), k_name)
            
            k_impact_abs = 0.0
            pct_change = 0.0
            peak_period_label = ""  # Переменная для хранения года максимального эффекта
            
            if k_id in sim_res:
                pct_change = sim_res[k_id].get('pct_change', 0.0)
                
                # Достаем точные данные по годам из симуляции (чтобы синхронизировать с таблицей)
                annual_data = sim_res[k_id].get('annual', {})
                if annual_data:
                    # Ищем год с максимальным абсолютным отклонением
                    max_dev = 0.0
                    max_yr = None
                    for yr, vals in annual_data.items():
                        plan = float(vals.get('plan', 0.0))
                        forecast = float(vals.get('forecast', plan))
                        dev = forecast - plan
                        if abs(dev) > abs(max_dev):
                            max_dev = dev
                            max_yr = yr
                            
                    k_impact_abs = max_dev
                    # Если год найден и есть хоть какое-то отклонение, формируем приписку
                    if max_yr is not None and abs(max_dev) > 1e-6:
                        peak_period_label = f"<br><span style='font-size:11px; color:gray'>Макс. эффект: {max_yr} год</span>"
                else:
                    # Резервный вариант, если годовых данных нет
                    k_node = engine.G.nodes.get(k_id, {})
                    k_base_val = safe_float(k_node.get('Year', 0))
                    k_impact_abs = pct_change * k_base_val
            
            # Добавляем год в название (работает благодаря поддержке HTML-тегов в Plotly)
            k_label = f"Δ {k_title_wrapped}{peak_period_label}"
            
            # Подстраиваем лимит шкалы под полученное значение
            g_limit = max(0.1, math.ceil(abs(k_impact_abs) * 1.5 * 10) / 10)
            
            _live_sim[k_name] = {
                'impact_abs': k_impact_abs, 
                'm': 1.0 + pct_change, 
                'share': k_share, 
                'label': k_label,
                'g_limit': g_limit
            }

    # Разделяем интерфейс на 3 независимые колонки
    col_sliders, col_bars, col_kpis = st.columns([0.8, 1.2, 1.3])

    with col_sliders:
        if active_years:
            for y_str in active_years:
                st.markdown(f"**Перенос для {y_str} года:**")

                key_req = f"ch_trans_req_{selected_entity}_{y_str}"
                key_add = f"ch_trans_add_{selected_entity}_{y_str}"
                
                # Заставляем Streamlit моментально обновить таблицу при движении мышкой
                def _force_rerun():
                    pass

                val_req = st.slider(
                    f"Канал 7 · Потребность → База, % ({y_str})", 0, 100, key=key_req,
                    on_change=_force_rerun,
                    help="Какую долю запрошенной потребности этого года считать переведённой "
                         "в базу (гарантированные средства).")
                val_add = st.slider(
                    f"Канал 8 · Доп. потребность → База, % ({y_str})", 0, 100, key=key_add,
                    on_change=_force_rerun,
                    help="Какую долю дополнительной потребности этого года считать переведённой в базу.")

                orig_r = next((r for r in _entity_data if r["Год"] == y_str), None)
                if orig_r:
                    amt_req = _fnum(orig_r.get("_orig_req")) * (val_req / 100.0)
                    amt_add = _fnum(orig_r.get("_orig_add")) * (val_add / 100.0)
                    
                    # --- УМНОЖАЕМ НА КОЭФФИЦИЕНТ БАЗЫ ---
                    eff_added = (amt_req + amt_add) * base_pct
                    
                    if eff_added > 0:
                        st.markdown(f"<div style='font-size:0.75rem; color:var(--mx-pos); margin-top:-10px; margin-bottom:10px;'>"
                                    f"✓ В Базу: +{fmt(eff_added)} {budget_unit}</div>", unsafe_allow_html=True)
        else:
            st.info("Включите год галочкой над таблицей, чтобы открыть ползунки переноса.")

    with col_bars:
        _dk = theme_is_dark()
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(x=plot_years, y=plot_base, name="База", marker_color="#36D399" if _dk else "#0E8F5E"))
        fig_bars.add_trace(go.Bar(x=plot_years, y=plot_req, name="Потребн.", marker_color="#E0A24A" if _dk else "#F5A623"))
        fig_bars.add_trace(go.Bar(x=plot_years, y=plot_add, name="Доп.", marker_color="#FF6B81" if _dk else "#D8425A"))
        fig_bars.add_trace(go.Scatter(x=plot_years, y=plot_feff, name="F_eff", mode="lines+markers",
                                      line=dict(color="#1B4DFF", width=3, dash='dot'), marker=dict(size=8)))

        fig_bars.update_layout(
            barmode='stack', height=340, margin=dict(l=10, r=10, t=50, b=10),
            title=dict(text=f"Реальный бюджет (дисконт.):<br><b>{fmt(total_feff)}</b> {budget_unit}", font=dict(size=14)),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        fig_bars.update_xaxes(showgrid=False)
        fig_bars.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

        st.plotly_chart(fig_bars, use_container_width=True, key=f"bars_{selected_entity}")

    with col_kpis:
        n_kpis = len(infl) if infl else 1

        if n_kpis <= 2:
            n_rows = len(infl) if infl else 1
            fig_kpi = make_subplots(rows=n_rows, cols=1, specs=[[{"type": "indicator"}]] * n_rows, vertical_spacing=0.1)

            if not infl:
                impact = ((V_val - V_orig) / V_orig) * 100.0 if V_orig > 0 else 0.0
                g_color = "#36D399" if impact >= 0 else "#FF6B81" if _dk else ("#0E8F5E" if impact >= 0 else "#D8425A")
                fig_kpi.add_trace(go.Indicator(
                    mode="gauge+number", value=round(impact, 2),
                    title={"text": "Δ Ценности (оценка)", "font": {"size": 13, "color": "gray"}},
                    number={'valueformat': '+.2f', 'suffix': '%', 'font': {'size': 24, 'color': g_color}},
                    gauge={'axis': {'range': [-100, 100]}, 'bar': {'color': g_color}}
                ), row=1, col=1)
            else:
                for idx, k_info in enumerate(infl):
                    k_name = k_info['KPI']
                    k_share = k_info['Влияние']

                    if k_name in _live_sim:
                        ls = _live_sim[k_name]
                        k_impact = ls['impact_abs']
                        k_label = ls['label']
                        g_limit = ls['g_limit']
                    else:
                        k_impact = 0.0
                        k_label = f"{k_name}<br>оценка"
                        g_limit = 1.0

                    g_color = "#36D399" if k_impact >= 0 else "#FF6B81" if _dk else ("#0E8F5E" if k_impact >= 0 else "#D8425A")

                    fig_kpi.add_trace(go.Indicator(
                        mode="gauge+number", value=round(k_impact, 3),
                        title={"text": k_label, "font": {"size": 12, "color": "gray"}},
                        number={'valueformat': '+.2f', 'suffix': '', 'font': {'size': 24, 'color': g_color}},
                        gauge={
                            'axis': {'range': [-g_limit, g_limit], 'tickwidth': 1, 'tickcolor': "gray"},
                            'bar': {'color': g_color, 'thickness': 0.75}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0,
                            'steps': [{'range': [-g_limit, 0], 'color': "rgba(216, 66, 90, 0.1)"},
                                      {'range': [0, g_limit], 'color': "rgba(14, 143, 94, 0.1)"}],
                            'threshold': {'line': {'color': "gray", 'width': 2}, 'thickness': 0.75, 'value': 0}
                        }
                    ), row=idx+1, col=1)

            fig_kpi.update_layout(
                height=340, margin=dict(l=20, r=20, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_kpi, use_container_width=True, key=f"gauges_{selected_entity}")

        else:
            chart_height = max(340, n_kpis * 85)
            k_names, k_impacts, k_colors, k_texts = [], [], [], []

            for k_info in reversed(infl):
                k_name = k_info['KPI']
                k_share = k_info['Влияние']

                if k_name in _live_sim:
                    ls = _live_sim[k_name]
                    k_impact = ls['impact_abs']
                    k_full_label = ls['label']
                else:
                    k_impact = 0.0
                    k_full_label = k_name

                wrapped_name = "<br>".join(textwrap.wrap(k_full_label, width=42))
                k_names.append(wrapped_name)
                k_impacts.append(k_impact)
                k_colors.append("#36D399" if k_impact >= 0 else "#FF6B81" if _dk else ("#0E8F5E" if k_impact >= 0 else "#D8425A"))
                k_texts.append(f" {k_impact:+.2f} ")
                
            fig_kpi = go.Figure()
            fig_kpi.add_trace(go.Bar(
                x=k_impacts, y=k_names,
                orientation='h',
                marker_color=k_colors,
                text=k_texts,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Изменение: %{x:+.2f}<extra></extra>"
            ))

            fig_kpi.update_layout(
                height=chart_height, margin=dict(l=10, r=40, t=50, b=10),
                title=dict(text="Живой прогноз KPI (абс. значение)", font=dict(size=14)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_kpi.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=True, zerolinecolor="gray", zerolinewidth=1)
            fig_kpi.update_yaxes(showgrid=False, automargin=True)

            st.plotly_chart(fig_kpi, use_container_width=True, key=f"tornado_{selected_entity}")

    # --- 4. DEBOUNCE ОТКЛЮЧЕН (УСТРАНЕНИЕ ОШИБКИ ФРАГМЕНТА) ---
    # now = time.time()
    # timer_key = f"_trans_timer_{selected_entity}"
    # if trans_changed:
    #     st.session_state[timer_key] = now
    # else:
    #     last_timer = st.session_state.get(timer_key, 0)
    #     if last_timer > 0 and (now - last_timer) > 0.8:
    #         st.session_state[timer_key] = 0
    #         st.session_state.table_nonce += 1
    #         # УБРАН st.rerun()

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Итог. Реальный бюджет (дисконт.)", fmt(total_feff))
    m2.metric("Дедлайн", new_end.strftime("%d.%m.%Y"))
    
def render_midi_setup_panel():
    """Панель настройки MIDI-пульта в sidebar: статус, привязки каналов, журнал событий."""
    if not MIDI_AVAILABLE:
        st.info("🎛️ MIDI-пульт не установлен — управление с экрана.")
        return

    midi_state = get_midi_state()
    port = midi_state.get('port')
    mapping = midi_state.get('mapping', [])
    events = midi_state.get('events', [])

    with st.expander("🎛️ MIDI-пульт", expanded=True):
        # Статус подключения
        if port:
            st.success(f"🟢 Подключён: {port}")
        else:
            st.warning("🔴 Пульт не обнаружен — переподключение через 3 сек…")

        # Таблица каналов
       # Обновленные списки без 5 канала
        channel_names = ["1: БАЗА", "2: ПОТРЕБ", "3: ДОП", "4: СДВИГ", "5: ТР.ПОТРЕБ", "6: ТР.ДОП"]
        ch_keys = ['base', 'req', 'add', 'shift', 'trans_req', 'trans_add']
        ch_min = [0, 0, 0, -12, 0, 0]
        ch_max = [100, 100, 0, 24, 100, 100]

        st.markdown("**Каналы пульта:**")
        for i in range(6): # <--- Было 7, стало 6
            with st.container():
                cc1, cc2, cc3 = st.columns([2, 2, 1])
                key = ch_keys[i]
                val = midi_state.get(key, ch_min[i])
                bound = mapping[i] if i < len(mapping) else "— привязан"
                with cc1:
                    st.markdown(f"**{channel_names[i]}**")
                with cc2:
                    st.caption(f"Сигнал: `{bound}` · Значение: {val}")
                with cc3:
                    if st.button("🔄", key=f"unbind_{i}", use_container_width=True):
                        if i < len(mapping):
                            del mapping[i]
                            midi_state['mapping'] = mapping
                            st.rerun()

        # Журнал событий
        if events:
            with st.expander("📋 Журнал событий", expanded=False):
                for ev in events[-10:]:
                    st.caption(ev)

        # Кнопка сброса
        if mapping:
            if st.button("🗑 Забыть все привязки", use_container_width=True):
                midi_state['mapping'] = []
                midi_state['cc_modes'] = {}
                st.rerun()



_MODEL_WIDGET_KEYS = {
    "mdl_alpha": "alpha", "mdl_lambda": "lambda_f", "mdl_beta": "beta", "mdl_sigk": "sigmoid_k",
    "mdl_bonus": "time_bonus_enabled", "mdl_late": "late_finish_penalty_enabled",
    "mdl_latew": "late_finish_weight", "mdl_agg": "default_agg_mode", "mdl_rho": "default_ces_rho",
    "mdl_actthr": "activation_threshold", "mdl_actk": "sigmoid_activation_k",
    "mdl_timep": "time_penalty_power", "mdl_fmode": "forecast_mode",
}


def sync_model_config():
    """Переносит значения виджетов вкладки «Модель» в конфиг ДО сборки движка.

    Вкладка рендерится после сборки, поэтому без этой синхронизации правка коэффициента
    применялась бы с задержкой в одно действие (классическая ловушка порядка выполнения
    в Streamlit: виджет уже вернул новое значение, но код сборки его ещё не прочитал)."""
    C = st.session_state.get("custom_config")
    if C is None:
        return
    for wkey, field in _MODEL_WIDGET_KEYS.items():
        if wkey in st.session_state:
            try:
                setattr(C, field, st.session_state[wkey])
            except Exception:
                pass


def render_model_tab(ctx, factor, budget_unit_name):
    """Вкладка «Модель»: калибровка математики. Вынесена из боковой панели — здесь есть ширина
    под графики и формулы, а редкая калибровка отделена от ежедневной работы со сценарием."""
    C = st.session_state.custom_config

    section("Настройка модели ценности", "как деньги и сроки превращаются в ценность работы")
    st.caption("Меняются редко — обычно один раз при внедрении. Изменения применяются сразу, "
               "ядро пересобирается на лету. Формулы и графики ниже показывают, что именно вы крутите.")

    # ── 1. ДЕНЬГИ ────────────────────────────────────────────────────────────
    st.markdown('<div class="mx-h"><span class="t">1 · Отдача от денег</span>'
                '<span class="s">· сколько ценности приносит бюджет</span></div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns([1, 1.25])
    with m1:
        cfg_alpha = st.slider(
            "Максимальная отдача от денег (α)", 0.5, 5.0, float(C.alpha), 0.1, key="mdl_alpha",
            help="«Потолок» кривой: какую ценность работа способна дать при очень большом бюджете. "
                 "Больше α — деньги для этого проекта важнее."
        )
        _eff = float(C.lambda_f) * factor
        _knee_now = 1.0 / max(_eff, 1e-12)
        cfg_lambda = st.slider(
            "Скорость насыщения (λ)", 0.001, 0.5, float(min(max(C.lambda_f, 0.001), 0.5)), 0.001,
            format="%g", key="mdl_lambda",
            help="Насколько быстро деньги перестают помогать. Больше λ — насыщение наступает раньше "
                 "(«колено» кривой сдвигается влево)."
        )
        _eff_new = cfg_lambda * factor
        st.markdown(
            f'<div style="font-size:.82rem;color:var(--mx-muted);">Насыщение начинается примерно после '
            f'<b style="color:var(--mx-ink);">{fmt(1.0 / max(_eff_new, 1e-12), 0)} {budget_unit_name}</b> '
            f'на работу — дальше каждый рубль даёт заметно меньше.</div>',
            unsafe_allow_html=True)
        st.latex(rf"V_{{\text{{деньги}}}} = {cfg_alpha:.2f} \cdot \ln(1 + {_eff_new:.4g} \cdot F_{{\text{{реал}}}})")

    with m2:
        f_max = max(50.0, (1.0 / max(_eff_new, 1e-12)) * 3.5)
        f_vals = np.linspace(0, f_max, 160)
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=f_vals, y=cfg_alpha * np.log1p(_eff_new * f_vals),
                                   mode='lines', line=dict(color='#1B4DFF', width=3),
                                   hovertemplate=f'Бюджет: %{{x:.0f}} {budget_unit_name}<br>Ценность: %{{y:.2f}}<extra></extra>'))
        fig_b.add_vline(x=1.0 / max(_eff_new, 1e-12), line_dash="dash", line_color="gray",
                        annotation_text=" насыщение")
        fig_b.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title=f"Бюджет работы ({budget_unit_name})", yaxis_title="Ценность",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_b, use_container_width=True, key="mdl_budget")

    st.divider()

    # ── 2. СРОКИ ─────────────────────────────────────────────────────────────
    st.markdown('<div class="mx-h"><span class="t">2 · Влияние сроков</span>'
                '<span class="s">· цена опоздания и премия за опережение</span></div>',
                unsafe_allow_html=True)

    t1, t2 = st.columns([1, 1.25])
    with t1:
        cfg_beta = st.slider(
            "Ценность попадания в срок (β)", 0.5, 6.0, float(C.beta), 0.25, key="mdl_beta",
            help="Сколько ценности даёт сам факт выполнения работы вовремя. Больше β — сроки важнее денег."
        )
        cfg_sig_k = st.slider(
            "Жёсткость сроков (k)", 0.5, 5.0, float(C.sigmoid_k), 0.25, key="mdl_sigk",
            help="Насколько резко падает ценность при отклонении от срока. Больше k — «жёсткий дедлайн»: "
                 "небольшая просрочка уже дорого стоит."
        )
        cb1, cb2 = st.columns(2)
        with cb1:
            cfg_bonus = st.checkbox("Премия за опережение", value=bool(C.time_bonus_enabled), key="mdl_bonus",
                                    help="Досрочное завершение немного повышает ценность.")
        with cb2:
            cfg_late = st.checkbox("Штраф за срыв срока", value=bool(C.late_finish_penalty_enabled), key="mdl_late",
                                   help="Дополнительно наказывает за сам факт срыва дедлайна.")
        cfg_late_w = st.slider(
            "Размер штрафа за срыв", 0.0, 1.5, float(C.late_finish_weight), 0.05,
            disabled=not cfg_late, key="mdl_latew",
            help="Какую долю от «ценности срока» можно потерять при сильном опоздании."
        )

    with t2:
        t_vals = np.linspace(0.0, 2.5, 160)
        eff_t = t_vals.copy()
        if not cfg_bonus:
            eff_t[eff_t < 1.0] = 1.0
        v_t = cfg_beta * (1.0 / (1.0 + np.exp(cfg_sig_k * (eff_t - 1.0))))
        if cfg_late:
            late_ratio = np.maximum(0, t_vals - 1.0)
            v_t -= cfg_late_w * cfg_beta * (2.0 * (1.0 / (1.0 + np.exp(-cfg_sig_k * late_ratio))) - 1.0)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_vals, y=v_t, mode='lines', line=dict(color='#D8425A', width=3),
                                   hovertemplate='Срок: %{x:.2f}× плана<br>Ценность: %{y:.2f}<extra></extra>'))
        fig_t.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text=" точно в срок")
        fig_t.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Фактический срок (1.0 = точно по плану)", yaxis_title="Ценность",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t, use_container_width=True, key="mdl_time")

    st.divider()

    # ── 3. СВЁРТКА ВВЕРХ ─────────────────────────────────────────────────────
    st.markdown('<div class="mx-h"><span class="t">3 · Сборка ценности вверх по графу</span>'
                '<span class="s">· как вклады работ соединяются в показатель</span></div>',
                unsafe_allow_html=True)

    a1, a2 = st.columns([1, 1.25])
    with a1:
        _agg_modes = {'ces': 'Современный (CES)', 'классический': 'Классический (с порогом)'}
        _agg_keys = list(_agg_modes.keys())
        cur_agg = getattr(C, 'default_agg_mode', 'ces')
        cfg_agg = st.selectbox(
            "Способ сборки", _agg_keys, key="mdl_agg",
            index=_agg_keys.index(cur_agg) if cur_agg in _agg_keys else 0,
            format_func=lambda k: _agg_modes[k],
            help="CES — гибкая свёртка, форма настраивается параметром ρ (и индивидуально во вкладке "
                 "«Экспертная настройка»). Классический — единый порог: слабые ветки не проходят наверх."
        )

        # ρ ПО УМОЛЧАНИЮ — раньше этого коэффициента в панели не было вовсе
        cfg_ces_rho = st.slider(
            "Форма свёртки по умолчанию (ρ)", -5.0, 1.0,
            float(getattr(C, 'default_ces_rho', 1.0)), 0.1, disabled=(cfg_agg != 'ces'), key="mdl_rho",
            help="ρ → 1: вклады складываются (работы взаимозаменяемы). ρ → 0: перемножаются (нужны все). "
                 "ρ → −5: берётся минимум — узкое место решает всё."
        )
        _rho_hint = ("вклады складываются — работы взаимозаменяемы" if cfg_ces_rho > 0.6 else
                     "нужны все работы — слабое звено тянет вниз" if cfg_ces_rho > -1.0 else
                     "решает узкое место (минимум)")
        st.caption(f"Сейчас: {_rho_hint}.")

        # ПОРОГ АКТИВАЦИИ — раньше показывался только текстом, без возможности изменить
        cfg_act_thr = st.slider(
            "Порог прохождения ценности", 0.0, 1.5,
            float(getattr(C, 'activation_threshold', 0.5)), 0.05,
            disabled=(cfg_agg != 'классический'), key="mdl_actthr",
            help="Ценность ниже порога не проходит наверх (шумовой фильтр). "
                 "Работает только в классическом режиме."
        )
        cfg_act_k = st.slider(
            "Резкость порога", 0.1, 3.0,
            float(getattr(C, 'sigmoid_activation_k', 1.0)), 0.1,
            disabled=(cfg_agg != 'классический'), key="mdl_actk",
            help="Чем выше, тем больше порог похож на жёсткую ступеньку."
        )
        if cfg_agg == 'ces':
            st.info("В режиме CES порог не используется — форму свёртки задаёт ρ.")

    with a2:
        if cfg_agg == 'классический':
            av = np.linspace(0.0, 1.5, 160)
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=av, y=1.0 / (1.0 + np.exp(-cfg_act_k * (av - cfg_act_thr))),
                                       mode='lines', line=dict(color='#0E8F5E', width=3),
                                       hovertemplate='Сырая ценность: %{x:.2f}<br>Пропускается: %{y:.2f}<extra></extra>'))
            fig_a.add_vline(x=cfg_act_thr, line_dash="dash", line_color="gray", annotation_text=" порог")
            fig_a.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Ценность, пришедшая снизу", yaxis_title="Доля, прошедшая наверх",
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_a, use_container_width=True, key="mdl_act")
        else:
            # наглядно: как ρ меняет свёртку двух работ
            x = np.linspace(0.05, 2.0, 120)
            fig_r = go.Figure()
            for rho, name, col in ((1.0, "ρ = 1 (сумма)", "#1B4DFF"),
                                   (0.0, "ρ → 0 (произведение)", "#6D3FD1"),
                                   (-4.0, "ρ = −4 (минимум)", "#D8425A")):
                other = 1.0
                if abs(rho) < 1e-6:
                    y = np.sqrt(x * other)
                else:
                    y = (0.5 * x ** rho + 0.5 * other ** rho) ** (1.0 / rho)
                fig_r.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name,
                                           line=dict(color=col, width=3 if abs(rho - cfg_ces_rho) < 0.6 else 1.5,
                                                     dash=None if abs(rho - cfg_ces_rho) < 0.6 else 'dot')))
            fig_r.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Ценность первой работы (вторая = 1.0)",
                                yaxis_title="Ценность родителя",
                                legend=dict(orientation="h", y=-0.3),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_r, use_container_width=True, key="mdl_rho_chart")

    st.divider()

    # ── 4. АВТО-БАЛАНСИРОВКА ─────────────────────────────────────────────────
    st.markdown('<div class="mx-h"><span class="t">4 · Авто-балансировка сроков</span>'
                '<span class="s">· насколько растягивается работа при урезании бюджета</span></div>',
                unsafe_allow_html=True)

    e1, e2 = st.columns([1, 1.25])
    with e1:
        cfg_time_p = st.slider(
            "Сила растяжения сроков", 0.25, 2.0, float(C.time_penalty_power), 0.05, key="mdl_timep",
            help="1.0 — линейно: урезали бюджет вдвое, срок вырос вдвое. "
                 "0.5 — мягко (корень). 2.0 — катастрофически быстро."
        )
        _ex = 2.0 ** cfg_time_p
        st.markdown(f'<div style="font-size:.85rem;color:var(--mx-muted);">Пример: урезали бюджет '
                    f'<b style="color:var(--mx-ink);">в 2 раза</b> → срок вырастет '
                    f'<b style="color:var(--mx-ink);">в {_ex:.2f} раза</b>.</div>', unsafe_allow_html=True)
        st.latex(r"\text{Срок}_{\text{нов}} = \text{Срок}_{\text{план}} \cdot "
                 rf"\left( \frac{{F_{{\text{{было}}}}}}{{F_{{\text{{стало}}}}}} \right)^{{{cfg_time_p:.2f}}}")
        st.caption("Работает только при УРЕЗАНИИ бюджета и только если включён тумблер ⚖️ в консоли. "
                   "Перенос денег между годами — это другой механизм (кассовый разрыв).")

    with e2:
        cut = np.linspace(1.0, 8.0, 160)
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=cut, y=cut ** cfg_time_p, mode='lines',
                                   line=dict(color='#F5A623', width=3),
                                   hovertemplate='Урезали в %{x:.1f} раз<br>Срок вырос в %{y:.2f} раз<extra></extra>'))
        fig_e.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Во сколько раз урезан бюджет",
                            yaxis_title="Во сколько раз вырос срок",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_e, use_container_width=True, key="mdl_ext")

    st.divider()

    # ── 5. ОТОБРАЖЕНИЕ ───────────────────────────────────────────────────────
    st.markdown('<div class="mx-h"><span class="t">5 · Отображение прогноза</span></div>',
                unsafe_allow_html=True)
    _fmodes = {'value': 'В единицах показателя', 'completion': 'В доле выполнения плана'}
    _fkeys = list(_fmodes.keys())
    cfg_fmode = st.selectbox(
        "Как показывать прогноз", _fkeys, key="mdl_fmode",
        index=_fkeys.index(getattr(C, 'forecast_mode', 'value')) if getattr(C, 'forecast_mode', 'value') in _fkeys else 0,
        format_func=lambda k: _fmodes[k],
        help="В абсолютных единицах показателя либо как процент выполнения плана."
    )

    # ── ПРИМЕНЕНИЕ ───────────────────────────────────────────────────────────
    C.alpha = cfg_alpha
    C.lambda_f = cfg_lambda
    C.beta = cfg_beta
    C.sigmoid_k = cfg_sig_k
    C.time_bonus_enabled = cfg_bonus
    C.late_finish_penalty_enabled = cfg_late
    C.late_finish_weight = cfg_late_w
    C.default_agg_mode = cfg_agg
    C.default_ces_rho = cfg_ces_rho
    C.activation_threshold = cfg_act_thr
    C.sigmoid_activation_k = cfg_act_k
    C.time_penalty_power = cfg_time_p
    C.forecast_mode = cfg_fmode

    st.divider()
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        if st.button("♻️ Вернуть значения по умолчанию", use_container_width=True):
            st.session_state.custom_config = mc.MixerConfig()
            st.rerun()
    with rc2:
        st.caption("Сброс возвращает все коэффициенты к заводским значениям "
                   "(α = 2.0, λ = 0.01, β = 1.5, k = 2.0, ρ = 1.0).")


def page_mixer():
    st.markdown("""
    <style>
    /* Слайдеры пульта — компактные горизонтальные, подпись прячем (label_visibility=collapsed) */
    div[data-testid="stSlider"] > label { display: none !important; }
    div[data-testid="stSlider"] { margin: 15px auto 5px auto; width: 90%; }

    /* Дизайн физической дорожки пульта M-Vave — адаптивные цвета */
    .mixer-channel {
        background: linear-gradient(180deg, var(--mx-surface-2), var(--mx-bg));
        border: 2px solid var(--mx-border);
        border-radius: 8px;
        padding: 10px 2px;
        text-align: center;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.06), 0 4px 6px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        align-items: center;
        height: 140px;
    }
    /* Экранчик с названием канала */
    .ch-label {
        font-family: var(--mx-mono); font-size: 0.55rem; color: var(--mx-pos);
        background: var(--mx-bg); padding: 3px 1px; border-radius: 3px;
        margin-bottom: 8px; box-shadow: inset 0 0 5px rgba(14,143,94,0.15);
        white-space: nowrap; overflow: hidden; font-weight: 700; width: 100%;
    }
    /* Индикатор текущего значения под ползунком */
    .ch-val {
        font-family: var(--mx-mono); font-size: 0.7rem; color: var(--mx-accent);
        font-weight: bold; margin-top: -3px; text-shadow: 0 0 3px rgba(27,77,255,0.3);
    }
    /* Заголовки пульта */
    .mx-h { 
        font-size: 1.1rem; 
        font-weight: 700; 
        color: #1f2937; 
        margin: 20px 0 10px 0; 
        padding-bottom: 5px;
        border-bottom: 2px solid #e5e7eb;
    }
    /* Метрики в стиле дашборда */
    [data-testid="stMetricValue"] { font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)
    apply_theme()  # #9: тёмная тема
    # --- Активный проект и изоляция состояния ---
    projects = ps.discover(ps.PROJECTS_ROOT)
    slug = st.session_state.get('active_project')
    ctx = ps.context_for(slug)
    if st.session_state.get('_loaded_project') != slug:
        for _k in ('selected_entity_id', 'simulation_results', 'ai_reports',
                   'scenario_params', 'sensitivity', 'realloc'):
            st.session_state.pop(_k, None)
        st.session_state['_loaded_project'] = slug
    st.session_state.setdefault('project_cfg', {})
    if slug not in st.session_state.project_cfg:
        _cfg = MixerConfig()
        if slug:  # подтягиваем сохранённую единицу бюджета проекта с диска
            try:
                _cfg.budget_scale = ps.load_project_settings(ctx).get('budget_scale', 'millions')
            except Exception:
                pass
        st.session_state.project_cfg[slug] = _cfg
    st.session_state.custom_config = st.session_state.project_cfg[slug]

    with st.sidebar:
        _names = {c.title: c.slug for c in projects}
        _titles = list(_names.keys())
        _idx = list(_names.values()).index(slug) if slug in _names.values() else 0
        _pick = st.selectbox("📁 Активный проект", _titles, index=_idx, key="proj_switcher")
        if _names.get(_pick) and _names[_pick] != slug:
            st.session_state.active_project = _names[_pick]
            st.rerun()
        if st.button("← Все проекты", use_container_width=True):
            st.session_state.active_project = None
            st.rerun()
        render_midi_setup_panel()
        st.divider()
        _ls = ps.load_llm_settings()
        _lp = ps.LLM_PRESETS.get(_ls.get('provider'), {}).get('label', _ls.get('provider'))
        if _ls.get('enabled'):
            st.caption(f"🤖 ИИ: {_lp} · {_ls.get('model')}")
        else:
            st.caption("🤖 ИИ выключен — расчёт по модели и кешам")
        st.divider()

    st.markdown(f"# 🎛️ {ctx.title}")
    st.markdown(
        '<div class="mx-sub">Что произойдёт с годовыми KPI, если изменить бюджет или сроки задачи. '
        'Выберите задачу, сдвиньте параметры — увидите прогноз ещё до утверждения.</div>',
        unsafe_allow_html=True,
    )

    # ======================================================================
    # БОКОВАЯ ПАНЕЛЬ — только КОНТЕКСТ проекта и допущения сценария.
    # Настройка математики модели вынесена во вкладку «Модель»: в узкой колонке
    # графики и формулы нечитаемы, а смешение «ежедневной работы» и «редкой
    # калибровки» — классическая ошибка информационной архитектуры.
    # ======================================================================
    if 'custom_config' not in st.session_state:
        st.session_state.custom_config = MixerConfig()
    _C = st.session_state.custom_config

    st.sidebar.markdown("### 💰 Деньги проекта")

    _scales = {'millions': 'Миллионы ₽', 'hundred_thousands': 'Сотни тысяч ₽',
               'thousands': 'Тысячи ₽', 'rub': 'Рубли'}
    _skeys = list(_scales.keys())
    current_scale = getattr(_C, 'budget_scale', 'millions')
    cfg_scale = st.sidebar.selectbox(
        "Единица бюджета в файлах", _skeys,
        index=_skeys.index(current_scale) if current_scale in _skeys else 0,
        format_func=lambda k: _scales[k],
        help="В каких единицах записаны суммы в вашем Excel. Влияет только на отображение и масштаб денег."
    )
    factor = mc.BUDGET_UNITS.get(cfg_scale, mc.BUDGET_UNITS['millions'])['factor']
    if cfg_scale != current_scale:
        _C.budget_scale = cfg_scale
        try:
            ps.save_project_settings(ctx, {'budget_scale': cfg_scale})
        except Exception:
            pass
        st.rerun()

    st.sidebar.markdown("### 🎲 Допущения сценария")
    st.sidebar.caption("Насколько мы верим, что запрошенные деньги действительно придут.")

    cfg_rho_req = st.sidebar.slider(
        "Потребность будет профинансирована, %", 0, 100,
        int(getattr(_C, 'rho_req', 1.0) * 100), 5,
        help="Потребность — это ЗАПРОШЕННЫЕ, но ещё не гарантированные деньги. "
             "100% — уверены, что дадут. 0% — считаем, что не дадут вовсе."
    ) / 100.0
    cfg_rho_add = st.sidebar.slider(
        "Доп. потребность будет профинансирована, %", 0, 100,
        int(getattr(_C, 'rho_add', 0.0) * 100), 5,
        help="Дополнительная потребность — резервный запрос сверх основного. Обычно 0%: на него не рассчитывают."
    ) / 100.0
    cfg_discount = st.sidebar.slider(
        "Ставка дисконтирования (инфляция), % в год", 0, 20,
        int(getattr(_C, 'discount_rate', 0.06) * 100), 1,
        help="Деньги, полученные позже, стоят меньше сегодняшних. Именно эта ставка приводит "
             "будущие суммы к базовому году. Обычно 6–10%."
    ) / 100.0
    cfg_base_year = st.sidebar.number_input(
        "Базовый год приведения", min_value=2000, max_value=2100,
        value=int(getattr(_C, 'base_year', 2026)), step=1,
        help="Год, к которому приводятся деньги при дисконтировании. Суммы этого года берутся "
             "без скидки, более поздние — со скидкой. Обычно это первый год проекта."
    )

    _C.rho_req = cfg_rho_req
    _C.rho_add = cfg_rho_add
    _C.discount_rate = cfg_discount
    _C.base_year = int(cfg_base_year)

    st.sidebar.caption("⚙️ Математика модели (веса бюджета и сроков, форма свёртки) — "
                       "во вкладке **«Модель»** на странице.")


    st.sidebar.markdown("### 💾 Данные и кеш")
    use_weight_cache = st.sidebar.checkbox("Загружать матрицу весов", value=True,
                                           help="Не дёргать LLM, если структура задач не менялась.")
    use_meth_cache = st.sidebar.checkbox("Загружать сжатые методики", value=True)
    force_recompress = st.sidebar.checkbox("Пересжать методики заново", value=False)

    if st.sidebar.button("🔄 Перезагрузить из Excel", use_container_width=True):
        get_build_manager().invalidate(slug)
        for _k in ('simulation_results','ai_reports','scenario_params','sensitivity','realloc'):
            st.session_state.pop(_k, None)
        st.rerun()

    if ctx.has_baseline():
        st.sidebar.caption("У проекта есть утверждённый базовый план.")
        if st.sidebar.button("♻️ Сбросить к плану из Excel", use_container_width=True,
                             help="Удалить утверждённый базовый план и вернуться к исходным данным Excel."):
            ps.reset_baseline(ctx)
            get_build_manager().invalidate(slug)
            for _k in ('simulation_results','ai_reports','scenario_params','sensitivity','realloc'):
                st.session_state.pop(_k, None)
            st.rerun()

    # ======================================================================
    # ИНИЦИАЛИЗАЦИЯ ЯДРА
    # ======================================================================
    sync_model_config()          # значения вкладки «Модель» должны попасть в конфиг ДО сборки
    cfg = st.session_state.custom_config
    mc.set_log_project(slug, ctx.log_path)
    flags = {'use_w_cache': use_weight_cache, 'use_m_cache': use_meth_cache, 'force': force_recompress}
    mgr = get_build_manager()
    sig = _build_signature(slug, ctx, cfg, flags)
    cfg_snapshot = MixerConfig(**vars(cfg))
    state = mgr.get(slug)
    if not state or state.get('signature') != sig:
        mgr.request(slug, sig,
                    lambda cb, _s=slug, _f=flags, _c=cfg_snapshot: _build_engine(_s, _f, _c, cb))
        state = mgr.get(slug)

    if state and state['status'] == 'error':
        st.error(f"🚨 Не удалось собрать систему: {state.get('error')}")
        st.info("Проверьте файлы проекта на вкладке «Проекты» и при необходимости загрузите заново.")
        st.stop()
    if not state or state['status'] != 'ready':
        render_loading_page(slug, ctx, mgr)
        return
    engine, llm = state['engine'], state['llm']

    st.session_state.setdefault('selected_entity_id', None)
    st.session_state.setdefault('simulation_results', None)
    st.session_state.setdefault('ai_reports', {})
    st.session_state.setdefault('scenario_params', None)
    st.session_state.setdefault('table_nonce', 0)

    kpi_ids = list(engine.kpi_ids)
    sim = st.session_state.simulation_results

    def kpi_year_values(k_id):
        node = engine.G.nodes[k_id]
        base_year = safe_float(node.get('Year', 0))
        if sim and k_id in sim:
            forecast_year = safe_float(sim[k_id]['quarters'].get('Год', {}).get('forecast', base_year))
            pct = sim[k_id].get('pct_change', 0.0)
        else:
            forecast_year, pct = base_year, 0.0
        return base_year, forecast_year, pct

    # ======================================================================
    # ВЕРХНЯЯ ПОЛОСА — ИТОГ ПО KPI
    # ======================================================================
    section("Годовые показатели", "верхнеуровневый итог по каждому KPI")
    if not kpi_ids:
        st.info("В данных нет ни одного KPI — проверьте файл плановых показателей.")
    else:
        cols = st.columns(min(len(kpi_ids), 4))
        for i, k_id in enumerate(kpi_ids):
            node = engine.G.nodes[k_id]
            base_year, forecast_year, pct = kpi_year_values(k_id)
            per = []
            try:
                per = json.loads(node['periods']) if isinstance(node.get('periods'), str) else (node.get('periods') or [])
            except Exception:
                per = []
            years = sorted({int(p['year']) for p in per}) if per else []
            yr_lbl = str(years[0]) if years else ""
            with cols[i % len(cols)]:
                trend = "up" if (sim and pct > CHANGE_THRESHOLD) else ("down" if (sim and pct < -CHANGE_THRESHOLD) else "")
                if sim:
                    delta_abs = forecast_year - base_year
                    sign = "+" if delta_abs >= 0 else "−"
                    body = (
                        f'<div class="row"><span class="big">{fmt(forecast_year)}</span>{pct_badge(pct)}</div>'
                        f'<div class="dlt">{sign}{fmt(abs(delta_abs))} к плану</div>'
                        f'{meter_html(base_year, forecast_year)}'
                    )
                    lbl = f"Прогноз{(' · ' + yr_lbl) if yr_lbl else ' (год)'}"
                else:
                    body = (f'<div class="row"><span class="big">{fmt(base_year)}</span></div>'
                            f'<div class="plan">сценарий не запущен</div>')
                    lbl = f"План{(' · ' + yr_lbl) if yr_lbl else ' (год)'}"
                extra = ""
                if len(years) > 1:
                    if sim and k_id in sim and sim[k_id].get('annual'):
                        ann = {int(y): v for y, v in sim[k_id]['annual'].items()}
                    else:
                        ann = engine._kpi_annual(k_id)
                    yr_rows = ""
                    for y in years:
                        a = ann.get(y, {})
                        pl = safe_float(a.get('plan', 0.0))
                        fc = safe_float(a.get('forecast', pl))
                        cls = 'pos' if (sim and fc > pl + 1e-9) else ('neg' if (sim and fc < pl - 1e-9) else '')
                        arrow = f'{fmt(pl)} → <b class="{cls}">{fmt(fc)}</b>' if sim else f'<b>{fmt(pl)}</b>'
                        yr_rows += f'<div class="yr"><span class="y">{y}</span><span class="pf">{arrow}</span></div>'
                    extra = f'<div class="mx-years">{yr_rows}</div>'
                if sim:
                    conf = float(engine.get_kpi_calibration(k_id).get('confidence', 1.0))
                    if conf < 0.999:
                        band = (1.0 - conf) * abs(forecast_year - base_year)
                        if band > 0:
                            extra += (f'<div class="plan">коридор: {fmt(forecast_year - band)} … '
                                      f'{fmt(forecast_year + band)} (уверенность {int(conf*100)}%)</div>')
                st.markdown(
                    f'<div class="mx-card mx-kpi {trend}"><div class="lbl">{lbl}</div>'
                    f'<div class="name">{node.get("name","")}</div>{body}{extra}</div>',
                    unsafe_allow_html=True,
                )

    ribbon = [chip(f"{len(kpi_ids)} KPI", "info")]
    if engine.budget_discrepancies:
        ribbon.append(chip(f"{len(engine.budget_discrepancies)} расхождени(е/я) бюджета", "warn"))
    if engine.schedule_violations:
        total_v = sum(len(v) for v in engine.schedule_violations.values())
        ribbon.append(chip(f"{total_v} календарн. нестыковк(а/и)", "bad"))
    if not engine.budget_discrepancies and not engine.schedule_violations:
        ribbon.append(chip("структура согласована", "ok"))
    st.markdown(f'<div class="mx-ribbon">{"".join(ribbon)}</div>', unsafe_allow_html=True)
    if engine.budget_discrepancies or engine.schedule_violations:
        st.caption("Подробности по расхождениям и нестыковкам — на вкладке «Чувствительность и бюджет»; "
                   "проблемные строки бюджета подсвечены в план-графике.")

    st.divider()

    # ======================================================================
    # ВКЛАДКИ
    # ======================================================================
    tab_scenario, tab_reverse, tab_analytics, tab_model, tab_expert, tab_unmatched = st.tabs(
        ["🎚️ Сценарий", "🎯 Обратный расчёт", "📈 Аналитика",
         "⚙️ Модель", "🧮 Веса связей", "💸 Несвязанные финансы"]
    )

    with tab_model:
        render_model_tab(ctx, factor, mc.budget_unit(st.session_state.custom_config.budget_scale))

    # ----------------------------------------------------------------------
    # ВКЛАДКА 1 — СЦЕНАРИЙ
    # ----------------------------------------------------------------------
    with tab_scenario:
        with st.expander("❓ Как это работает — за 30 секунд", expanded=False):
            hw1, hw2, hw3 = st.columns(3)
            with hw1:
                st.markdown("**1 · Выберите задачу**")
                st.caption("Нажмите на строку в план-графике. Откроется микшер: деньги по годам, "
                           "сроки и вероятности финансирования.")
            with hw2:
                st.markdown("**2 · Измените деньги или сроки**")
                st.caption("Правки видны сразу — таблица и бюджеты пересчитываются на лету. "
                           "Пульт M-Vave меняет те же параметры фейдерами.")
            with hw3:
                st.markdown("**3 · Примените сценарий**")
                st.caption("Показатели пересчитаются по кварталам. Эффект включается с квартала "
                           "завершения работы. Ничего не утверждается, пока вы не нажмёте «Утвердить».")
            st.caption("Работа, завершившаяся в прошлом, меняет и прошедшие кварталы — это ретроспектива "
                       "(«что было бы, если»). Фактические значения при этом не подменяются.")

        section("План-график", "кликните по строке, чтобы открыть микшер задачи")

        f1, f2 = st.columns([1.6, 1])
        
        # Оставили только поиск, убрали радио-кнопки фильтров
        query = st.text_input("Поиск по наименованию или номеру работы", value="",
                              placeholder="например: внедрение или 1.2",
                              key="main_search")

        type_icon = {'веха': '◆', 'мероприятие': '▸', 'подзадача': '▪', 'задача': '■'}

        def _display_finances(node_id):
            """Финансы узла ДЛЯ ОТОБРАЖЕНИЯ: у листа — эффективные (свои + цель родителей);
            у родителя — агрегат ПО ГОДАМ/СТАТУСАМ эффективных финансов его листьев (свой
            источник родителя может быть пуст/очищен после правок). Единый источник для
            таблицы и микшера, чтобы цифры не расходились."""
            attr_ = engine.G.nodes[node_id]
            is_leaf_ = engine.G.in_degree(node_id) == 0
            if is_leaf_:
                return mc.ProjectMixer._parse_finances(
                    attr_.get('finances_eff', attr_.get('finances', {})))
            agg_ = {}
            for lf in engine._leaf_descendants(node_id):
                lfin = mc.ProjectMixer._parse_finances(
                    engine.G.nodes[lf].get('finances_eff', engine.G.nodes[lf].get('finances', {})))
                for y_str, amounts in lfin.items():
                    acc = agg_.setdefault(y_str, {'base': 0.0, 'req_extra': 0.0, 'add': 0.0})
                    for st_ in ('base', 'req_extra', 'add'):
                        acc[st_] += float(amounts.get(st_, 0.0) or 0.0)
            return agg_

        all_finance_years = set()
        for n, attr in engine.G.nodes(data=True):
            if str(attr.get('type', '')).upper() == 'KPI': continue
            fin_data = _display_finances(n)
            all_finance_years.update(int(y) for y in fin_data.keys())
        sorted_years = sorted(list(all_finance_years))

        rows = []
        for n, attr in engine.G.nodes(data=True):
            if str(attr.get('type', '')).upper() == 'KPI':
                continue
            
            # Убрали фильтрацию по признаку "Листовые" и "С расхождением"
            if query:
                q = query.strip().lower()
                if q not in str(n).lower() and q not in str(attr.get('name', '')).lower():
                    continue
                    
            t = str(attr.get('type', '')).strip()
            icon = type_icon.get(t.lower(), '•')
            
            fin_data = _display_finances(n)

            # Получаем текущие проценты из боковой панели
            cfg_rho_req = getattr(st.session_state.custom_config, 'rho_req', 1.0)
            cfg_rho_add = getattr(st.session_state.custom_config, 'rho_add', 0.0)

            # Считаем профиль, обеспеченность и эффективный бюджет
            yearly_totals = []
            total_base = 0.0
            total_req = 0.0
            total_add = 0.0
            eff_budget = 0.0
            
            for yr in sorted_years:
                yr_data = fin_data.get(str(yr), {})
                b = float(yr_data.get('base', 0.0))
                r = float(yr_data.get('req_extra', 0.0))
                a = float(yr_data.get('add', 0.0))
                
                yearly_totals.append(b + r + a)
                total_base += b
                total_req += r
                total_add += a
                
                eff_budget += b + (r * cfg_rho_req) + (a * cfg_rho_add)
                
            # Умный расчет обеспеченности
            if (total_base + total_req) > 0:
                coverage = (total_base / (total_base + total_req)) * 100.0
            elif total_add > 0:
                coverage = 0.0
            else:
                coverage = 100.0 if float(attr.get('F', 0.0)) > 1e-9 else 0.0

            if eff_budget <= 1e-9 and float(attr.get('F', 0.0)) > 1e-9:
                eff_budget = float(attr.get('F', 0.0))

            # Формируем строку без колонок "Расхождение" и "Питает KPI"
            row_dict = {
                'id': n,
                'Тип': f"{icon} {t}",
                'Наименование': attr.get('name', ''),
                'V': round(safe_float(attr.get('agg_value', 0.0)), 2),
                'Начало': '' if mc.ProjectMixer._is_milestone_type(t) else fmt_date(attr.get('T_start', '')),
                'Конец': fmt_date(attr.get('T_end', '')),
                'Бюджет': round(eff_budget, 2),
                'Обеспеченность': coverage,
                'Профиль': yearly_totals,
                'Фин. веха': bool(attr.get('is_financial', False)),
            }
                
            rows.append(row_dict)

        def _skey(i):
            return [int(p) if str(p).isdigit() else p for p in str(i).split('.')]

        SHOW_COLS = ['id', 'Тип', 'Наименование', 'V', 'Начало', 'Конец', 'Бюджет', 'Обеспеченность', 'Профиль', 'Фин. веха']

        if not rows:
            st.info("Под фильтр ничего не подошло — измените поиск.")
        else:
            ordered = sorted(rows, key=lambda r: _skey(r['id']))
            available_cols = [c for c in SHOW_COLS if all(c in row for row in ordered)]
            df_g = pd.DataFrame(ordered)[available_cols]
            unit = mc.budget_unit(engine.config.budget_scale)
            
            prof_title = f'Профиль ({min(sorted_years)}–{max(sorted_years)})' if sorted_years else 'Профиль'

            dfcfg = {
                'id': st.column_config.TextColumn('№', width='small'),
                'Наименование': st.column_config.TextColumn('Наименование', width='large'),
                'V': st.column_config.NumberColumn('V', format='%.2f'),
                'Бюджет': st.column_config.NumberColumn(f'Бюджет, {unit}', format='%.2f'),
                'Обеспеченность': st.column_config.ProgressColumn('База, %', min_value=0, max_value=100, format='%d%%'),
                'Профиль': st.column_config.BarChartColumn(prof_title, y_min=0),
                'Фин. веха': st.column_config.CheckboxColumn(
                    'Фин. веха', width='small',
                    help='Финансовая сущность: деньги родителей распределяются только на такие '
                         'вехи (по весу). Переключается в микшере задачи; включается само при вводе денег.'),
            }

            ev = st.dataframe(
                df_g, width="stretch", hide_index=True, height=430,
                column_config=dfcfg,
                selection_mode="single-row", on_select="rerun",
                key=f"sched_df_{st.session_state.table_nonce}",
            )
            sel_rows = []
            try:
                sel_rows = ev.selection.rows
            except Exception:
                sel_rows = (ev.get('selection', {}) or {}).get('rows', []) if isinstance(ev, dict) else []
            if sel_rows and sel_rows[0] < len(ordered):
                chosen = str(ordered[sel_rows[0]]['id'])
                if chosen in engine.G.nodes and chosen != st.session_state.selected_entity_id:
                    st.session_state.selected_entity_id = chosen
                    st.rerun()
            st.caption(f"Клик по строке открывает микшер задачи · сортировка по столбцам · бюджет в «{unit}». "
                       "Кнопка ниже — скачать; разворот на весь экран — по наведению на правый верх таблицы.")
            
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
                df_excel = df_g.drop(columns=['Профиль'], errors='ignore')
                df_excel.to_excel(w, index=False)
            st.download_button("⬇ Скачать таблицу (Excel)", data=buf.getvalue(), file_name="план-график.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_sched", use_container_width=True)
        selected_entity = st.session_state.selected_entity_id

        # --- МИКШЕР выбранной задачи ---
        col_mix, col_proj = st.columns([1.1, 1])
        with col_mix:
            if selected_entity and selected_entity in engine.G.nodes:
                node = engine.G.nodes[selected_entity]
                cur_name = node.get('name', '')
                cur_type = node.get('type', '')
                old_f = safe_float(node.get('F', 0.0))
                kids = list(engine.G.predecessors(selected_entity))
                real_kids = [c for c in kids if str(engine.G.nodes[c].get('type', '')).upper() != 'KPI']
                is_leaf = len(real_kids) == 0
                tag = ('<span class="mx-pill mx-tag-leaf">листовая</span>' if is_leaf
                       else '<span class="mx-pill mx-tag-parent">родительская</span>')

                section("Микшер задачи")
                head_l, head_r = st.columns([5, 1])
                head_l.markdown(f'**[{selected_entity}] {cur_name}** {tag}', unsafe_allow_html=True)
                if head_r.button("Закрыть", use_container_width=True):
                    st.session_state.selected_entity_id = None
                    st.session_state.simulation_results = None
                    st.session_state.ai_reports = {}
                    st.session_state.scenario_params = None
                    st.session_state.table_nonce += 1
                    # --- ОЧИСТКА ТАБЛИЦЫ ---
                    st.session_state.pop(f"finance_profile_{selected_entity}", None)
                    st.rerun()

                with st.container(border=True):
                    if not is_leaf:
                        st.caption("ℹ️ Это родительская задача: ИЗМЕНЕНИЕ бюджета (Δ) распределяется "
                                   "между непосредственными подзадачами ПРОПОРЦИОНАЛЬНО ВЕСУ их влияния; "
                                   "текущие бюджеты подзадач сохраняются (при неизменной сумме ничего не двигается).")

                    infl = engine.task_kpi_influences(selected_entity)
                    if len(infl) > 1:
                        st.markdown(
                            '<span style="font-size:.8rem;color:var(--mx-muted);">Влияет на показатели:</span> ' +
                            "".join(
                                f'<span class="mx-chip info"><span class="dot"></span>'
                                f'{r["KPI"]}: {r["Влияние"]*100:.0f}%</span> '
                                for r in infl
                            ) +
                            '<span style="font-size:.72rem;color:var(--mx-faint);" '
                            'title="У каждого показателя свой знаменатель, поэтому проценты не складываются в 100%.">'
                            ' ⓘ проценты не суммируются</span>',
                            unsafe_allow_html=True)
                    elif len(infl) == 1:
                        st.markdown(
                            '<span style="font-size:.8rem;color:var(--mx-muted);">Влияет на показатель:</span> '
                            f'<span class="mx-chip info"><span class="dot"></span>'
                            f'{infl[0]["KPI"]}: {infl[0]["Влияние"]*100:.0f}%</span>',
                            unsafe_allow_html=True)

                    
                    auto_balance = st.checkbox("⚖️ Авто-балансировка: при урезании бюджета продлевать сроки", value=True, key=f"auto_balance_{selected_entity}")
                    _unit = mc.budget_unit(engine.config.budget_scale)
                    
                    # === ЧАСТЬ 1: ФИНАНСОВАЯ МАТРИЦА ===
                    st.markdown('<div class="mx-h"><span class="t">1. Финансовый профиль</span><span class="s">· ввод данных</span></div>', unsafe_allow_html=True)
                    
                    fin_data = _display_finances(selected_entity)

                    # --- ФИНАНСОВАЯ СУЩНОСТЬ (управление флагом; в таблице план-графика — индикатор) ---
                    if is_leaf:
                        _cur_fl = bool(node.get('is_financial', False))
                        _new_fl = st.checkbox(
                            "💰 Финансовая веха (участвует в распределении денег родителей)",
                            value=_cur_fl, key=f"finflag_{selected_entity}",
                            help="Деньги, заданные на родительских задачах, распределяются только "
                                 "между финансовыми вехами — пропорционально их весу. "
                                 "При вводе денег вручную флаг включается автоматически.")
                        if _new_fl != _cur_fl:
                            node['is_financial'] = _new_fl
                            try:
                                _st_ = ps.load_project_settings(ctx)
                                _st_.setdefault('financial_flags', {})[selected_entity] = bool(_new_fl)
                                ps.save_project_settings(ctx, _st_)
                            except Exception:
                                pass
                            engine.recalculate()   # перераспределить деньги с учётом нового состава
                            st.session_state.table_nonce += 1
                            st.rerun()

                    years = sorted([int(y) for y in fin_data.keys()]) if fin_data else [2026, 2027, 2028]
                    if not years: years = [2026, 2027, 2028]
                    start_y, end_y = min(years), max(years)
                    if end_y < start_y + 2: end_y = max(end_y, start_y + 2)

                    state_key = f"finance_profile_{selected_entity}"
                    if state_key not in st.session_state:
                        rows_init = []
                        for y in range(start_y, end_y + 1):
                            d = fin_data.get(str(y), {})
                            rows_init.append({
                                "Год": str(y),
                                "База": float(d.get("base", 0.0)),
                                "Потребность": float(d.get("req_extra", 0.0)),
                                "Доп. потребность": float(d.get("add", 0.0)),
                                "_orig_base": float(d.get("base", 0.0)), # добавлено для консистентности
                                "_orig_req": float(d.get("req_extra", 0.0)),
                                "_orig_add": float(d.get("add", 0.0))
                            })
                        st.session_state[state_key] = rows_init

                    st.markdown('<div class="mx-h"><span class="t">Перераспределение средств (Каналы 7 и 8)</span></div>', unsafe_allow_html=True)
                    st.caption("Выберите годы для настройки переноса. Ползунки и график появятся на пульте ниже.")
                    
                    years_list = [r["Год"] for r in st.session_state[state_key]]
                    chk_cols = st.columns(len(years_list))
                    active_years = []
                    for idx, y_str in enumerate(years_list):
                        with chk_cols[idx]:
                            if st.checkbox(f"Настроить {y_str}", key=f"yr_active_{selected_entity}_{y_str}"):
                                active_years.append(y_str)

                    if active_years:
                        if len(active_years) > 1:
                            st.radio("🎯 Фокус пульта (Каналы 7 и 8) направлен на год:", 
                                     options=active_years, 
                                     key=f"midi_focus_year_{selected_entity}", 
                                     horizontal=True)
                        else:
                            st.session_state[f"midi_focus_year_{selected_entity}"] = active_years[0]
                    
                    # === СИНХРОНИЗАЦИЯ ПЕРЕНОСА ПЕРЕД ОТРИСОВКОЙ ТАБЛИЦЫ ===
                    _apply_transfer_deltas(selected_entity, active_years, state_key)
                    # =======================================================
                    
                    # Отдаем в таблицу данные из сессии без системных колонок _orig
                    df_fin = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in st.session_state[state_key]])
                    
                    def on_fin_edit():
                        # Когда вы вносите цифры руками, они сразу сохраняются в мастер-слепок
                        midi_s = get_midi_state()
                        editor_key = f"fin_editor_{selected_entity}_{st.session_state.table_nonce}"
                        edits = st.session_state[editor_key].get("edited_rows", {})
                        
                        for row_idx, changes in edits.items():
                            y_str = st.session_state[state_key][row_idx]["Год"]
                            for col, val in changes.items():
                                # Удаление значения в ячейке даёт None — трактуем как 0.0
                                try:
                                    float_val = float(val) if val is not None else 0.0
                                except (TypeError, ValueError):
                                    float_val = 0.0
                                if float_val < 0:
                                    float_val = 0.0
                                st.session_state[state_key][row_idx][col] = float_val
                                
                                # Обновляем скрытые поля и сбрасываем ползунки, в т.ч. в памяти пульта
                                if col == "Потребность":
                                    st.session_state[state_key][row_idx]["_orig_req"] = float_val
                                    st.session_state[f"ch_trans_req_{selected_entity}_{y_str}"] = 0
                                    st.session_state[f"_last_val_ch_trans_req_{selected_entity}_{y_str}"] = 0
                                    if MIDI_AVAILABLE: midi_s['trans_req'] = 0
                                elif col == "Доп. потребность":
                                    st.session_state[state_key][row_idx]["_orig_add"] = float_val
                                    st.session_state[f"ch_trans_add_{selected_entity}_{y_str}"] = 0
                                    st.session_state[f"_last_val_ch_trans_add_{selected_entity}_{y_str}"] = 0
                                    if MIDI_AVAILABLE: midi_s['trans_add'] = 0
                                elif col == "База":
                                    st.session_state[state_key][row_idx]["_orig_base"] = float_val

                    edited_fin = st.data_editor(
                        df_fin,
                        column_config={
                            "Год": st.column_config.TextColumn("Год", disabled=True),
                            "База": st.column_config.NumberColumn(f"База, {_unit}", min_value=0.0, format="%.2f"),
                            "Потребность": st.column_config.NumberColumn(f"Потребн., {_unit}", min_value=0.0, format="%.2f"),
                            "Доп. потребность": st.column_config.NumberColumn(f"Доп., {_unit}", min_value=0.0, format="%.2f")
                        },
                        hide_index=True,
                        use_container_width=True,
                        key=f"fin_editor_{selected_entity}_{st.session_state.table_nonce}",
                        on_change=on_fin_edit
                    )
                    
                    # Сохраняем итоговую таблицу для кнопки "Применить сценарий"
                    st.session_state[f"_tmp_edited_fin_{selected_entity}"] = edited_fin
                    
                    # === M-VAVE CONSOLE: изолированный фрагмент реального времени ===
                    cur_start = datetime.strptime(node['T_start'], "%Y-%m-%d").date()
                    cur_end = datetime.strptime(node['T_end'], "%Y-%m-%d").date()
                    is_ms = mc.ProjectMixer._is_milestone_type(node.get('type'))

                    def_rho_req = int(float(node.get('rho_req', 1.0)) * 100)
                    def_rho_add = int(float(node.get('rho_add', 0.0)) * 100)

                    _cfg = st.session_state.get('custom_config', None)
                    _disc_rate = float(getattr(_cfg, 'discount_rate', 0.06)) if _cfg is not None else 0.06
                    _base_year = int(getattr(_cfg, 'base_year', 2026)) if _cfg is not None else 2026
                    
                    realtime_console_fragment(
                        selected_entity, node, engine, def_rho_req, def_rho_add, 
                        cur_start, cur_end, is_ms, active_years, _unit, infl, 
                        auto_balance, _disc_rate, _base_year
                    )

                    # === КНОПКА ЗАПУСКА И ТУМБЛЕР ===
                    
                    
                    calc_mode = st.radio(
                        "Глубина расчёта",
                        options=["⚡ Core (Мгновенно)", "🤖 AI Engine (Анализ по кварталам)"],
                        horizontal=True,
                        help="Core даёт быстрый процентный прогноз. AI дополнительно перераспределяет эффект по кварталам для каждого затронутого KPI.",
                    )
                    run = st.button("🚀 Применить параметры и пересчитать прогноз", type="primary", use_container_width=True)

                if run:
                    # 1. Финальные значения пульта
                    btn_base_pct = st.session_state.get(f"ch_base_{selected_entity}", 100) / 100.0
                    btn_rho_req  = st.session_state.get(f"ch_req_{selected_entity}", int(float(node.get('rho_req', 1.0)) * 100)) / 100.0
                    btn_rho_add  = st.session_state.get(f"ch_add_{selected_entity}", int(float(node.get('rho_add', 0.0)) * 100)) / 100.0
                    btn_shift_m  = st.session_state.get(f"ch_shift_{selected_entity}", 0)

                    # 2. Собираем финансы
                    def _num(v):
                        # Удалённая ячейка матрицы приходит как None — безопасно приводим к 0.0
                        try:
                            f = float(v) if v is not None else 0.0
                        except (TypeError, ValueError):
                            f = 0.0
                        return f if f > 0 else 0.0
                    updated_fin_dict = {}
                    new_F = 0.0
                    # ВАЖНО: берём данные из st.session_state[state_key] (мастер-слепок), а НЕ
                    # из edited_fin (значение виджета data_editor). Перенос по Каналам 7/8
                    # выполняется внутри realtime_console_fragment, который перерисовывается
                    # САМ ПО СЕБЕ при перетаскивании ползунка — сам виджет таблицы вне фрагмента
                    # в этот момент не перерисовывается и хранит СТАРЫЙ снимок (ключ виджета не
                    # менялся), поэтому edited_fin.iterrows() отдавал бы значения ДО переноса.
                    for r in st.session_state[state_key]:
                        b_v = _num(r.get('База')) * btn_base_pct
                        r_v = _num(r.get('Потребность'))
                        a_v = _num(r.get('Доп. потребность'))
                        updated_fin_dict[str(r['Год'])] = {'base': b_v, 'req_extra': r_v, 'add': a_v}
                        new_F += b_v + (r_v * btn_rho_req) + (a_v * btn_rho_add)
                    
                    rho_req, rho_add = btn_rho_req, btn_rho_add

                    # 3. Сроки, Автобаланс и Честный Номинал
                    b_dur = max(1, (cur_end - cur_start).days)
                    c_start = cur_start + timedelta(days=btn_shift_m * 30)
                    slider_dur_btn = float(b_dur)
                    
                    # Восстанавливаем настоящий старый бюджет (считаем и реальный, и номинал!)
                    old_fin_dict_btn = {}
                    old_F_nominal_btn = 0.0
                    old_rho_req = float(node.get('rho_req', 1.0))
                    old_rho_add = float(node.get('rho_add', 0.0))

                    for r in st.session_state[f"finance_profile_{selected_entity}"]:
                        y_str = str(r['Год'])
                        o_b = _fnum(r.get('_orig_base'))
                        o_r = _fnum(r.get('_orig_req'))
                        o_a = _fnum(r.get('_orig_add'))
                        old_fin_dict_btn[y_str] = {'base': o_b, 'req_extra': o_r, 'add': o_a}
                        # Честно складываем старый номинал из таблицы!
                        old_F_nominal_btn += o_b + (o_r * old_rho_req) + (o_a * old_rho_add)
                    
                    _, F_new_real_btn, _ = engine._evaluate_node_finances(updated_fin_dict, rho_req=rho_req, rho_add=rho_add)
                    _, F_old_real_btn, _ = engine._evaluate_node_finances(old_fin_dict_btn, rho_req=old_rho_req, rho_add=old_rho_add)
                    
                    if auto_balance and F_old_real_btn > 0 and F_new_real_btn < F_old_real_btn:
                        penalty = float(getattr(st.session_state.custom_config, 'time_penalty_power', 1.0))
                        safe_f = max(F_new_real_btn, F_old_real_btn * 0.05)
                        final_dur_btn = slider_dur_btn * ((F_old_real_btn / safe_f) ** penalty)
                    else:
                        final_dur_btn = slider_dur_btn
                        
                    c_end = c_start + timedelta(days=max(1, int(final_dur_btn)))
                    if is_ms:
                        c_start = c_end 
                        
                    ns, ne = c_start.strftime("%Y-%m-%d"), c_end.strftime("%Y-%m-%d")

                    if ne < ns:
                        st.error("Дата окончания раньше даты начала — поправьте сроки.")
                    else:
                        # 4. Запускаем Core
                        res = engine.mix(selected_entity, new_F, ns, ne, new_finances=updated_fin_dict, rho_req=rho_req, rho_add=rho_add)
                        
                        st.session_state.scenario_params = {
                            "entity": selected_entity, "F": new_F, "start": ns, "end": ne,
                            "F_old": old_F_nominal_btn,  # <--- Теперь сюда идет правильная сумма, а не фейковые 629k
                            "start_old": node['T_start'], "end_old": node['T_end'],
                            "new_finances": updated_fin_dict,
                            "rho_req": rho_req,
                            "rho_add": rho_add,
                            "F_old_real": F_old_real_btn, 
                            "F_new_real": F_new_real_btn  
                        }
                        st.session_state.ai_reports = {}

                        affected = [k for k, v in res.items() if abs(v['pct_change']) > CHANGE_THRESHOLD]
                        if not affected:
                            st.session_state.simulation_results = res
                            st.toast("Изменение поглощено порогами активации — KPI не сдвинулись.")
                        elif "ИИ" in calc_mode:
                            reports = {}
                            def _entity_weight(kid):
                                try:
                                    parents = [p for p in engine.G.successors(selected_entity)]
                                    wd = engine.kpi_weights.get(kid, {})
                                    ws = [wd.get((selected_entity, p), {}).get('weight') for p in parents]
                                    ws = [float(w) for w in ws if w is not None]
                                    if not ws:
                                        w0 = wd.get((selected_entity, kid), {}).get('weight')
                                        return float(w0) if w0 is not None else None
                                    return sum(ws) / len(ws)
                                except Exception:
                                    return None
                            with st.spinner(f"ИИ определяет новые значения по кварталам для {len(affected)} показателя(ей)…"):
                                for k_id in affected:
                                    kn = engine.G.nodes[k_id]
                                    ai = llm.generate_impact_report(
                                        f"{cur_type} «{cur_name}»",
                                        kn.get('name'),
                                        res[k_id]['pct_change'],
                                        weight=_entity_weight(k_id),
                                        influence=res[k_id].get('share'),
                                        periods=res[k_id].get('periods', []),
                                        annual=res[k_id].get('annual', {}),
                                        meth_text=engine.methodologies.get(k_id, ""),
                                    )
                                    reports[k_id] = ai.get('text_report', '')
                                    ai_values = ai.get('values', {}) or {}
                                    periods = res[k_id].get('periods', [])
                                    direction = safe_float(res[k_id].get('pct_change', 0.0))

                                    for pp in periods:
                                        val = ai_values.get(pp['label'])
                                        val = float(val) if val is not None else float(pp['forecast'])
                                        if direction < -1e-9:
                                            val = min(val, pp['plan'])
                                        elif direction > 1e-9:
                                            val = max(val, pp['plan'])
                                        val = engine._clamp_kpi_forecast(k_id, val)
                                        pp['forecast'] = round(val, 4)
                                        pp['deviation'] = round(val - pp['plan'], 4)
                                        pp['changed'] = abs(val - pp['plan']) > 1e-9
                            st.session_state.ai_reports = reports
                            st.session_state.simulation_results = res
                        else:
                            st.session_state.simulation_results = res
                        st.rerun()
            else:
                st.markdown(
                    '<div class="mx-card" style="text-align:center;padding:26px 18px;">'
                    '<div style="font-size:1.6rem;line-height:1;">🎚️</div>'
                    '<div style="font-weight:700;margin-top:8px;">Выберите задачу в план-графике</div>'
                    '<div style="font-size:.85rem;color:var(--mx-muted);margin-top:4px;">'
                    'Нажмите на строку выше — откроется микшер: деньги по годам, сроки и вероятности '
                    'финансирования.</div></div>', unsafe_allow_html=True)
        with col_proj:
            section("Прогноз по KPI", "трёхдатная поквартальная логика")
            st.caption("Эффект — с квартала завершения работы и далее.")
            st.markdown(
                '<div class="mx-legend">'
                '<span><i style="background:var(--mx-border-2)"></i>план</span>'
                '<span><i style="background:var(--mx-accent)"></i>прогноз</span>'
                '<span><i style="background:var(--mx-pos-bg);border:1px solid var(--mx-pos)"></i>рост</span>'
                '<span><i style="background:var(--mx-neg-bg);border:1px solid var(--mx-neg)"></i>спад</span>'
                '<span><i style="background:var(--mx-surface-2);border:1px solid var(--mx-border)"></i>🔒 закрытый квартал (факт)</span>'
                '</div>', unsafe_allow_html=True)

            if not selected_entity:
                st.caption("Здесь появится прогноз показателей, как только вы выберете задачу "
                           "и примените сценарий.")
            else:
                affected_set = set()
                if sim:
                    affected_set = {k for k, v in sim.items() if abs(v.get('pct_change', 0)) > CHANGE_THRESHOLD}

                kset = set(kpi_ids)
                subtree = {selected_entity} | {a for a in nx.ancestors(engine.G, selected_entity) if a not in kset}
                relevant = set()
                for _t in subtree:
                    relevant |= {d for d in nx.descendants(engine.G, _t) if d in kset}
                for _p in engine.G.successors(selected_entity):
                    if _p not in kset:
                        relevant |= {d for d in nx.descendants(engine.G, _p) if d in kset}
                if selected_entity in kset:
                    relevant.add(selected_entity)
                kpi_ids_shown = [k for k in kpi_ids if k in relevant]
                if not kpi_ids_shown:
                    kpi_ids_shown = list(kpi_ids)
                    st.caption("Эта работа не связана напрямую ни с одним KPI — показаны все показатели.")
                else:
                    st.caption(f"Показаны только показатели, зависящие от выбранной работы и её родителя "
                               f"({len(kpi_ids_shown)} из {len(kpi_ids)}).")

                q_label = {1: 'I', 2: 'II', 3: 'III', 4: 'IV'}

                def periods_for(k_id):
                    if sim and k_id in sim and sim[k_id].get('periods'):
                        return sim[k_id]['periods']
                    raw = engine.G.nodes[k_id].get('periods')
                    base = []
                    try:
                        base = json.loads(raw) if isinstance(raw, str) else (raw or [])
                    except Exception:
                        base = []
                    today = datetime.now()
                    out = []
                    for p in base:
                        ps, pe = engine._quarter_dates(int(p['year']), int(p['q']))
                        locked = pe < today
                        plan = float(p.get('plan', 0.0)); fact = float(p.get('fact', 0.0))
                        fore = plan
                        out.append({'year': int(p['year']), 'q': int(p['q']),
                                    'plan': plan, 'forecast': fore, 'fact': fact,
                                    'deviation': 0.0, 'locked': locked, 'changed': False})
                    return out

                def annual_for(k_id):
                    if sim and k_id in sim and sim[k_id].get('annual'):
                        return {int(y): {'plan': float(v.get('plan', 0.0)),
                                         'forecast': float(v.get('forecast', 0.0))}
                                for y, v in sim[k_id]['annual'].items()}
                    base = engine._kpi_annual(k_id)
                    return {int(y): {'plan': float(v.get('plan', 0.0)),
                                     'forecast': float(v.get('plan', 0.0))}
                            for y, v in base.items()}

                all_years = sorted({p['year'] for k in kpi_ids_shown for p in periods_for(k)})
                focus = "Все годы"
                if len(all_years) > 1:
                    focus = st.radio("Показать год", ["Все годы"] + [str(y) for y in all_years],
                                     horizontal=True,
                                     help="Ограничить карточки прогноза одним годом.")

                for k_id in kpi_ids_shown:
                    node = engine.G.nodes[k_id]
                    is_aff = k_id in affected_set
                    title = node.get('name', k_id)
                    badge = " " + pct_badge(sim[k_id]['pct_change']) if (sim and k_id in sim) else ""
                    st.markdown(f"**{title}**{badge}" + ("" if is_aff or not sim else "  ·  _без изменений_"),
                                unsafe_allow_html=True)

                    periods = periods_for(k_id)
                    if not periods:
                        st.caption("Нет квартальных данных по этому показателю.")
                        st.markdown("")
                        continue

                    years = sorted({p['year'] for p in periods})
                    annual = annual_for(k_id)

                    ann_chips = []
                    for yr in years:
                        ann = annual.get(yr, {'plan': 0.0, 'forecast': 0.0})
                        ap, af = ann['plan'], ann['forecast']
                        kind = "ok" if af > ap + 0.01 else ("bad" if af < ap - 0.01 else "info")
                        ann_chips.append(chip(f"{yr}: {fmt(ap)} → {fmt(af)}", kind))
                    st.markdown(f'<div class="mx-ribbon">{"".join(ann_chips)}</div>', unsafe_allow_html=True)

                    vis = [p for p in periods if (focus == "Все годы" or str(p['year']) == focus)]
                    vis = sorted(vis, key=lambda z: (z['year'], z['q']))

                    # Мини-график план vs прогноз по кварталам (СГРУППИРОВАННЫЕ столбцы).
                    chart_df = pd.DataFrame(
                        {"План": [p['plan'] for p in vis], "Прогноз": [p['forecast'] for p in vis]},
                        index=[f"{q_label.get(p['q'], p['q'])} {str(p['year'])[2:]}" for p in vis],
                    )
                    if not chart_df.empty:
                        _dk = theme_is_dark()
                        # Задаем цвета под активную тему (светлую/темную)
                        c_plan = "#3A414E" if _dk else "#C3C9D6"
                        c_pos = "#36D399" if _dk else "#0E8F5E"  # зеленый для роста
                        c_neg = "#FF6B81" if _dk else "#D8425A"  # красный для спада
                        c_neut = "#5B8CFF" if _dk else "#1B4DFF" # синий для без изменений
                        
                        plan_vals = chart_df["План"].tolist()
                        forecast_vals = chart_df["Прогноз"].tolist()
                        labels = chart_df.index.tolist()
                        
                        # Вычисляем цвет для каждого столбца "Прогноза" индивидуально
                        prog_colors = []
                        for p_val, f_val in zip(plan_vals, forecast_vals):
                            if f_val > p_val + 1e-4:
                                prog_colors.append(c_pos)
                            elif f_val < p_val - 1e-4:
                                prog_colors.append(c_neg)
                            else:
                                prog_colors.append(c_neut)
                                
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=labels, y=plan_vals, name='План', marker_color=c_plan,
                            hovertemplate='<b>%{x}</b><br>План: %{y:.2f}<extra></extra>'
                        ))
                        fig.add_trace(go.Bar(
                            x=labels, y=forecast_vals, name='Прогноз', marker_color=prog_colors,
                            hovertemplate='<b>%{x}</b><br>Прогноз: %{y:.2f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            barmode='group', 
                            height=240, 
                            margin=dict(l=0, r=0, t=10, b=0),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
                            xaxis=dict(showgrid=False, fixedrange=True),
                            yaxis=dict(showgrid=True, gridcolor="#262D38" if _dk else "#E2E7EE", fixedrange=True)
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # Таблица: кварталы по годам + годовой итог (восстановленный блок)
                    rows_kpi, flags_list = [], []
                    for yr in [y for y in years if (focus == "Все годы" or str(y) == focus)]:
                        yp = sorted([x for x in vis if x['year'] == yr], key=lambda z: z['q'])
                        for p in yp:
                            lock_mark = ""  # Убрали эмодзи "🔒 "
                            rows_kpi.append({"Период": f"{lock_mark}{q_label.get(p['q'], p['q'])} кв. {yr}",
                                 "План": p['plan'], "Прогноз": p['forecast'],
                                 "Факт": float(p['fact'] or 0.0), "Откл.": p['deviation']})
                            flags_list.append('locked' if p['locked'] else ('chg' if p['changed'] else ''))
                        ann = annual.get(yr, {'plan': 0.0, 'forecast': 0.0})
                        rows_kpi.append({"Период": f"Год {yr}", "План": ann['plan'], "Прогноз": ann['forecast'],
                                     "Факт": 0.0, "Откл.": ann['forecast'] - ann['plan']})
                        flags_list.append('year')

                    tdf = pd.DataFrame(rows_kpi)
                    
                    # Безопасный словарь для стилей, чтобы Pandas не падал при тестовых прогонах
                    flags_dict = dict(enumerate(flags_list))
                    _pal = table_palette()
                    
                    def style_row(row):
                        f = flags_dict.get(row.name, '')
                        n = len(row)
                        
                        if f == 'year': return [_pal['year']] * n
                        if f == 'locked': return [_pal['locked']] * n
                        
                        try:
                            d = float(row.get('Откл.', 0.0))
                        except Exception:
                            d = 0.0
                            
                        if d < -0.01: return [_pal['neg']] * n
                        if d > 0.01: return [_pal['pos']] * n
                        return [''] * n

                    styled_kpi = tdf.style.apply(style_row, axis=1).format({
                        "План": lambda v: fmt(v),
                        "Прогноз": lambda v: fmt(v),
                        "Факт": lambda v: fmt(0.0) if pd.isna(v) else fmt(v),
                        "Откл.": lambda v: ("+" if v >= 0 else "") + fmt(v),
                    })
                    st.dataframe(styled_kpi, width="stretch", hide_index=True)

                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
                        tdf.to_excel(w, sheet_name='KPI', index=False)
                    st.download_button(
                        "📥 Отчёт в Excel", data=buf.getvalue(),
                        file_name=f"KPI_{k_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True, key=f"dl_{k_id}",
                    )

                    if not (sim and k_id in sim) and st.session_state.ai_reports.get(k_id):
                        with st.expander("🤖 Заключение ИИ по рискам", expanded=is_aff):
                            st.markdown(st.session_state.ai_reports[k_id])

                    if sim and k_id in sim:
                        s = sim[k_id]
                        params = st.session_state.scenario_params or {}
                        work_id = params.get('entity')
                        work_nm = engine.G.nodes[work_id].get('name', work_id) if work_id and work_id in engine.G.nodes else "выбранная работа"
                        mult = 1.0 + s.get('pct_change', 0.0)
                        infl_pct = s.get('share', 0.0) * 100.0
                        direction = "вырос" if s.get('pct_change', 0.0) >= 0 else "снизился"
                        ex = next((p for p in periods if not p.get('locked') and abs(p['forecast'] - p['plan']) > 1e-6), None)
                        
                        _has_ai = bool(st.session_state.ai_reports.get(k_id))
                        _bd_labels = ["Простыми словами", "Формулы с числами"] + (["Заключение ИИ"] if _has_ai else [])
                        with st.expander("🔍 Разбор расчёта", expanded=is_aff and _has_ai):
                            _bd = st.tabs(_bd_labels)
                        with _bd[0]:
                            st.markdown(
                                f"1. Вы изменили работу **«{work_nm}»** (бюджет/срок).\n"
                                f"2. От этого поменялся её **вклад** в показатель. Сила связи этой работы "
                                f"с показателем — около **{infl_pct:.0f}%** (доля влияния).\n"
                                f"3. С учётом всех связей в графе показатель **{direction}** на "
                                f"**{abs(s.get('pct_change',0.0))*100:.1f}%**. Это и есть множитель "
                                f"**× {mult:.3f}**, на который умножается план.\n"
                                f"4. Каждый период пересчитывается как «план × {mult:.3f}».\n"
                                f"5. Эффект включается **с квартала завершения работы** и держится дальше."
                            )
                            if ex:
                                st.markdown(
                                    f"**Пример:** {ex['label']} — план {fmt(ex['plan'])} × {mult:.3f} = "
                                    f"**{fmt(ex['forecast'])}** (отклонение {('+' if ex['deviation']>=0 else '')}{fmt(ex['deviation'])})."
                                )
                            st.caption("Годовая строка считается так же — от отдельного годового плана, "
                                       "а не как сумма кварталов.")

                        with _bd[1]:
                            import math as _math
                            cfg = engine.config
                            a, lam_eff = cfg.alpha, engine._eff_lambda()
                            _u = mc.budget_unit(cfg.budget_scale)
                            
                            F_old_nominal = safe_float(params.get('F_old'))
                            F_new_nominal = safe_float(params.get('F'))
                            start_old = params.get('start_old'); end_old = params.get('end_old')
                            start_new = params.get('start'); end_new = params.get('end')
                            
                            work_node = engine.G.nodes.get(work_id, {}) if work_id else {}
                            is_ms_w = mc.ProjectMixer._is_milestone_type(work_node.get('type'))
                            
                            # Достаем заранее рассчитанные РЕАЛЬНЫЕ (с дисконтированием) цифры
                            F_old_real = params.get('F_old_real', 0.0)
                            F_new_real = params.get('F_new_real', 0.0)

                            def _val_F(F):
                                return a * _math.log1p(lam_eff * max(0.0, F))

                            if is_ms_w:
                                Vw_old = engine._milestone_value(work_id, F_old_real, end_old) if work_id else _val_F(F_old_real)
                                Vw_new = engine._milestone_value(work_id, F_new_real, end_new) if work_id else _val_F(F_new_real)
                                time_label = "срок (опоздание даты достижения)"
                            else:
                                try:
                                    d_old = max(1, (datetime.strptime(str(end_old)[:10], "%Y-%m-%d") - datetime.strptime(str(start_old)[:10], "%Y-%m-%d")).days)
                                    d_new = max(1, (datetime.strptime(str(end_new)[:10], "%Y-%m-%d") - datetime.strptime(str(start_new)[:10], "%Y-%m-%d")).days)
                                except Exception:
                                    d_old = d_new = 1
                                T_opt = safe_float(work_node.get('T_opt', d_old)) or d_old
                                late_old = engine._late_days(work_node, end_old) if work_id else 0
                                late_new = engine._late_days(work_node, end_new) if work_id else 0
                                Vw_old = engine._calculate_local_value(F_old_real, d_old, T_opt=T_opt, late_days=late_old)
                                Vw_new = engine._calculate_local_value(F_new_real, d_new, T_opt=T_opt, late_days=late_new)
                                time_label = "срок (отклонение длительности)"
                            
                            val_F_old, val_F_new = _val_F(F_old_real), _val_F(F_new_real)
                            val_T_old, val_T_new = Vw_old - val_F_old, Vw_new - val_F_new

                            Vk_old = safe_float(s.get('old')); Vk_new = safe_float(s.get('new'))
                            m_raw = (Vk_new / Vk_old) if Vk_old > 0 else 1.0

                            st.markdown(f"**1) Что поменяли** — работа «{work_nm}»:")
                            st.markdown(f"- Бюджет (номинал): {fmt(F_old_nominal)} → {fmt(F_new_nominal)} {_u}")
                            st.markdown(f"- Бюджет (реальный, дисконт. — идёт в ценность): "
                                        f"{fmt(F_old_real)} → {fmt(F_new_real)} {_u}")
                            if not is_ms_w:
                                st.markdown(f"- Срок (длительность): {d_old} дн. → {d_new} дн.")
                            else:
                                st.markdown(f"- Дата достижения: {fmt_date(end_old)} → {fmt_date(end_new)}")
                            st.markdown("**2) Ценность работы** (вклад в результат) = бюджетный член "
                                        f"+ временной член ({time_label}):")
                            st.latex(r"V_{\text{раб}} = \underbrace{\alpha\,\ln(1+\lambda_{\text{эфф}}\cdot F_{реальн.})}_{\text{бюджет}} "
                                     r"+ \underbrace{\beta\cdot\sigma(\dots)}_{\text{срок}}")
                            st.latex(rf"\text{{было}}:\ \underbrace{{{val_F_old:.3f}}}_{{\text{{бюджет}}}} + "
                                     rf"\underbrace{{{val_T_old:.3f}}}_{{\text{{срок}}}} = {Vw_old:.3f}")
                            st.latex(rf"\text{{стало}}:\ \underbrace{{{val_F_new:.3f}}}_{{\text{{бюджет}}}} + "
                                     rf"\underbrace{{{val_T_new:.3f}}}_{{\text{{срок}}}} = {Vw_new:.3f}")
                            if abs(val_T_new - val_T_old) > 1e-6:
                                st.caption(f"Временной член изменился ({val_T_old:.3f} → {val_T_new:.3f}) — "
                                           f"это и есть эффект от сдвига срока.")
                            st.markdown(f"**3) Ценность показателя** после прохода по графу "
                                        f"(агрегатор CES, влияние работы ≈ {infl_pct:.0f}%):")
                            st.latex(rf"V_{{\text{{KPI}}}}:\ {Vk_old:.3f}\ \rightarrow\ {Vk_new:.3f}")
                            st.markdown("**4) Множитель** (отношение ценностей) и калибровка к плану:")
                            st.latex(rf"m_{{\text{{сырое}}}} = \frac{{V_{{\text{{нов}}}}}}{{V_{{\text{{стар}}}}}} "
                                     rf"= \frac{{{Vk_new:.3f}}}{{{Vk_old:.3f}}} = {m_raw:.3f}")
                            st.latex(rf"m_{{\text{{калибр.}}}} = {mult:.3f}\quad(\text{{с учётом доверия и чувствительности}})")
                            st.markdown("**5) Прогноз периода** — план × m (для открытых периодов):")
                            if ex:
                                st.latex(rf"{ex['label']}:\ {ex['plan']:g}\times {mult:.3f} = {ex['forecast']:g}")
                            st.caption("λ_эфф зависит от выбранной единицы бюджета; «сырой» множитель — "
                                       "до калибровки, итоговый m — после сжатия к плану. В расчёте ценности "
                                       "используется РЕАЛЬНЫЙ (дисконтированный) бюджет, а не номинальный.")

                        if _has_ai:
                            with _bd[2]:
                                st.markdown(st.session_state.ai_reports[k_id])
                    st.markdown("")

                if sim:
                    st.divider()
                    c1, c2 = st.columns(2)
                    params = st.session_state.scenario_params or {}
                    with c1:
                        if st.button("✅ Утвердить сценарий", type="primary", use_container_width=True,
                                     help="Зафиксировать изменения как новый базовый план проекта (сохраняется на диск)."):
                            if params:
                                engine.commit(params["entity"], params["F"], params["start"], params["end"], sim, new_finances=params.get("new_finances"), rho_req=params.get("rho_req", 1.0), rho_add=params.get("rho_add", 0.0))
                                # авто-флаг «Финансовая веха» после ввода денег — сохранить (переживает пересборку)
                                try:
                                    if params.get("new_finances") and engine.G.nodes.get(params["entity"], {}).get('is_financial'):
                                        _st_ = ps.load_project_settings(ctx)
                                        _st_.setdefault('financial_flags', {})[params["entity"]] = True
                                        ps.save_project_settings(ctx, _st_)
                                except Exception:
                                    pass
                                ps.save_baseline(ctx, engine.export_state())
                            st.session_state.simulation_results = None
                            st.session_state.ai_reports = {}
                            st.session_state.scenario_params = None
                            # --- ОЧИСТКА ТАБЛИЦЫ ---
                            st.session_state.pop(f"finance_profile_{selected_entity}", None)
                            st.session_state.table_nonce += 1
                            
                            st.success("Сценарий зафиксирован как базовый план проекта.")
                            st.rerun()
                    with c2:
                        if st.button("↩️ Отменить сценарий", use_container_width=True,
                                     help="Вернуться к текущему базовому плану без изменений."):
                            st.session_state.simulation_results = None
                            st.session_state.ai_reports = {}
                            st.session_state.scenario_params = None
                            # --- ОЧИСТКА ТАБЛИЦЫ ---
                            st.session_state.pop(f"finance_profile_{selected_entity}", None)
                            st.session_state.table_nonce += 1
                            
                            st.rerun()

                    with st.expander("💾 Сохранить как сценарий (без утверждения)"):
                        sc_name = st.text_input("Название сценария", key="scenario_name",
                                                placeholder="например: ускорение Q4")
                        if st.button("Сохранить сценарий", key="save_scenario"):
                            if not (sc_name or "").strip():
                                st.warning("Укажите название сценария.")
                            else:
                                snap = engine.export_state()
                                if params:
                                    tmp = engine.export_state()
                                    engine.commit(params["entity"], params["F"], params["start"], params["end"], sim, new_finances=params.get("new_finances"), rho_req=params.get("rho_req", 1.0), rho_add=params.get("rho_add", 0.0))
                                # авто-флаг «Финансовая веха» после ввода денег — сохранить (переживает пересборку)
                                try:
                                    if params.get("new_finances") and engine.G.nodes.get(params["entity"], {}).get('is_financial'):
                                        _st_ = ps.load_project_settings(ctx)
                                        _st_.setdefault('financial_flags', {})[params["entity"]] = True
                                        ps.save_project_settings(ctx, _st_)
                                except Exception:
                                    pass
                                    snap = engine.export_state()
                                    engine.apply_state(tmp)
                                ps.save_scenario(ctx, sc_name.strip(), snap,
                                                 meta={"задача": params.get("entity"), "бюджет": params.get("F")})
                                st.success(f"Сценарий «{sc_name.strip()}» сохранён.")

            saved = ps.list_scenarios(ctx)
            if saved:
                st.divider()
                section("Сохранённые сценарии", "применить как базовый план или удалить")
                for s in saved:
                    sc1, sc2, sc3 = st.columns([3, 1, 1])
                    sc1.markdown(f"**{s['name']}**  ·  _{s['saved_at']}_")
                    if sc2.button("Применить", key=f"apply_{s['file']}", use_container_width=True):
                        engine.apply_state(s['state'])
                        ps.save_baseline(ctx, engine.export_state())
                        st.session_state.simulation_results = None
                        st.session_state.ai_reports = {}
                        st.success(f"Сценарий «{s['name']}» применён как базовый план.")
                        st.rerun()
                    if sc3.button("Удалить", key=f"delsc_{s['file']}", use_container_width=True):
                        ps.delete_scenario(ctx, s['file'])
                        st.rerun()

    # ----------------------------------------------------------------------
    # ВКЛАДКА 2 — ОБРАТНЫЙ РАСЧЁТ: цель по показателю → деньги и сроки
    # ----------------------------------------------------------------------
    with tab_reverse:
        section("Обратный расчёт", "задайте желаемое значение показателя — узнайте, сколько денег нужно или высвободится")
        st.caption("Прямой сценарий отвечает «что будет с показателем, если изменить деньги». "
                   "Здесь наоборот: вы задаёте цель по показателю, а модель считает необходимый бюджет "
                   "и показывает, как сдвинется план-график.")

        _unit_r = mc.budget_unit(engine.config.budget_scale)
        rk_left, rk_right = st.columns([1, 1.35])

        with rk_left:
            kpi_opts = {str(engine.G.nodes[k].get('name', k)): k for k in engine.kpi_ids}
            if not kpi_opts:
                st.info("В проекте нет показателей.")
            else:
                r_kpi_name = st.selectbox("Показатель", list(kpi_opts.keys()), key="rev_kpi")
                r_kpi = kpi_opts[r_kpi_name]

                r_periods = engine._kpi_periods(r_kpi)
                if not r_periods:
                    st.warning("У показателя нет квартальных плановых значений.")
                else:
                    now_ = datetime.now()
                    p_opts, p_map = [], {}
                    for p in r_periods:
                        y, q = int(p['year']), int(p['q'])
                        try:
                            _, q_end = engine._quarter_dates(y, q)
                            closed = q_end < now_
                        except Exception:
                            closed = False
                        lbl = f"{q} кв. {y}" + (" — прошедший" if closed else "")
                        p_opts.append(lbl)
                        p_map[lbl] = (y, q, float(p.get('plan', 0.0) or 0.0), closed)

                    # по умолчанию — первый открытый квартал
                    _def = next((i for i, l in enumerate(p_opts) if not p_map[l][3]), 0)
                    r_lbl = st.selectbox("Год и квартал", p_opts, index=_def, key="rev_period")
                    r_year, r_q, r_plan, r_closed = p_map[r_lbl]

                    if r_closed:
                        st.info("Квартал уже прошёл — это будет РЕТРОСПЕКТИВНЫЙ расчёт: "
                                "сколько денег понадобилось бы, чтобы показатель вышел на цель. "
                                "Фактические значения при этом не изменяются.")

                    st.markdown(f'<div class="mx-sub">Плановое значение квартала: '
                                f'<b style="color:var(--mx-ink);">{fmt(r_plan)}</b></div>',
                                unsafe_allow_html=True)

                    r_target = st.number_input(
                        "Целевое значение показателя", value=float(round(r_plan, 2)),
                        step=max(0.01, round(abs(r_plan) * 0.05, 2)) if r_plan else 1.0,
                        key="rev_target",
                        help="Больше плана — модель посчитает, сколько денег НУЖНО ДОБАВИТЬ. "
                             "Меньше плана — сколько денег ВЫСВОБОДИТСЯ.")

                    r_mode_lbl = st.radio(
                        "Как менять бюджет",
                        ["Распределить по всем влияющим работам", "Только одна работа"],
                        key="rev_mode", horizontal=False)
                    r_mode = 'proportional' if r_mode_lbl.startswith("Распределить") else 'single'

                    r_entity = None
                    cands, late = engine.target_candidates(r_kpi, r_year, r_q)
                    if r_mode == 'single':
                        if cands:
                            c_opts = {f"{c['id']} — {c['name'][:44]} (влияние {c['influence']*100:.0f}%)": c['id']
                                      for c in cands}
                            r_entity = c_opts[st.selectbox("Работа", list(c_opts.keys()), key="rev_entity")]
                        else:
                            st.warning("Нет подходящих работ (см. пояснение справа).")

                    run_rev = st.button("🎯 Рассчитать необходимый бюджет", type="primary",
                                        use_container_width=True)
                    if run_rev:
                        with st.spinner("Обратный расчёт…"):
                            st.session_state['rev_sol'] = engine.solve_for_target(
                                r_kpi, r_year, r_q, float(r_target), mode=r_mode, entity_id=r_entity)

        with rk_right:
            sol = st.session_state.get('rev_sol')
            if not sol:
                st.info("Выберите показатель, квартал и целевое значение — затем нажмите «Рассчитать».")

            elif not sol.get('feasible'):
                st.error(f"Цель недостижима: {sol.get('reason','')}")
                if sol.get('best_forecast') is not None:
                    st.caption(f"Максимально достижимое значение показателя в этом квартале: "
                               f"**{fmt(sol['best_forecast'])}** (дальше работает насыщение — "
                               f"каждый следующий рубль даёт всё меньше).")
                if sol.get('floor_forecast') is not None:
                    st.caption(f"Минимально достижимое значение (даже при нулевом бюджете этих работ): "
                               f"**{fmt(sol['floor_forecast'])}** — остальную часть дают другие работы.")
                if sol.get('late'):
                    st.markdown("**Работы, влияющие на показатель, но завершающиеся ПОЗЖЕ квартала:**")
                    st.caption("Деньги не ускоряют работу — их можно только сдвинуть по срокам во вкладке «Сценарий».")
                    st.dataframe(pd.DataFrame([{'№': w['id'], 'Работа': w['name'],
                                                'Влияние, %': round(w['influence'] * 100, 1),
                                                'Конец': fmt_date(w['end'])} for w in sol['late'][:8]]),
                                 hide_index=True, use_container_width=True)
            else:
                delta = sol['money_delta']
                if sol['direction'] == 'add':
                    st.markdown(
                        f'<div class="mx-card" style="border-left:4px solid var(--mx-neg);">'
                        f'<div style="font-size:.72rem;letter-spacing:.06em;color:var(--mx-muted);'
                        f'font-weight:700;">ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНО</div>'
                        f'<div style="font-family:var(--mx-display);font-size:2rem;font-weight:700;'
                        f'color:var(--mx-neg);font-variant-numeric:tabular-nums;">{fmt(abs(delta))} '
                        f'<span style="font-size:.5em;color:var(--mx-muted);">{_unit_r}</span></div>'
                        f'<div style="font-size:.82rem;color:var(--mx-muted);">бюджет работ: '
                        f'{fmt(sol["money_before"])} → {fmt(sol["money_after"])}</div></div>',
                        unsafe_allow_html=True)
                elif sol['direction'] == 'free':
                    st.markdown(
                        f'<div class="mx-card" style="border-left:4px solid var(--mx-pos);">'
                        f'<div style="font-size:.72rem;letter-spacing:.06em;color:var(--mx-muted);'
                        f'font-weight:700;">ВЫСВОБОЖДАЕТСЯ</div>'
                        f'<div style="font-family:var(--mx-display);font-size:2rem;font-weight:700;'
                        f'color:var(--mx-pos);font-variant-numeric:tabular-nums;">{fmt(abs(delta))} '
                        f'<span style="font-size:.5em;color:var(--mx-muted);">{_unit_r}</span></div>'
                        f'<div style="font-size:.82rem;color:var(--mx-muted);">бюджет работ: '
                        f'{fmt(sol["money_before"])} → {fmt(sol["money_after"])}</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.info("Цель совпадает с текущим прогнозом — менять бюджет не нужно.")

                if sol.get('retrospective'):
                    st.caption("↩︎ Ретроспектива: квартал уже прошёл. Расчёт показывает, какого бюджета "
                               "не хватило (или сколько было в избытке). Фактические значения не меняются.")

                c1, c2, c3 = st.columns(3)
                c1.metric("План квартала", fmt(sol['plan']))
                c2.metric("Цель", fmt(sol['target']),
                          delta=f"{sol['target'] - sol['plan']:+.2f}")
                c3.metric("Прогноз после", fmt(sol['forecast_after']),
                          delta=f"×{sol['m_achieved']:.3f}")

                st.markdown("**Деньги по работам**")
                st.dataframe(pd.DataFrame([{
                    '№': w['id'], 'Работа': w['name'][:46],
                    'Влияние, %': round(w['influence'] * 100, 1),
                    f'Было, {_unit_r}': w['before'], f'Стало, {_unit_r}': w['after'],
                    'Δ': w['delta'], 'Конец': fmt_date(w['end']),
                } for w in sol['per_work']]), hide_index=True, use_container_width=True)

                if sol.get('by_year'):
                    st.markdown("**Деньги по годам**")
                    st.dataframe(pd.DataFrame([{
                        'Год': y, f'Было, {_unit_r}': v['before'],
                        f'Стало, {_unit_r}': v['after'], 'Δ': v['delta'],
                    } for y, v in sorted(sol['by_year'].items())]), hide_index=True, use_container_width=True)

                if sol.get('schedule'):
                    st.markdown("**Изменения в плане-графике**")
                    st.dataframe(pd.DataFrame([{
                        '№': s['id'], 'Работа': s['name'][:46],
                        'Срок был': fmt_date(s['end_before']), 'Срок стал': fmt_date(s['end_after']),
                    } for s in sol['schedule']]), hide_index=True, use_container_width=True)
                else:
                    st.caption("План-график не сдвигается: деньги меняются внутри тех же лет, "
                               "кассовый разрыв не возникает.")

                if sol.get('late'):
                    with st.expander(f"Работы, которые деньгами не помогут ({len(sol['late'])})"):
                        st.caption("Они влияют на показатель, но завершаются позже целевого квартала. "
                                   "Деньги не ускоряют работу — нужен сдвиг сроков во вкладке «Сценарий».")
                        st.dataframe(pd.DataFrame([{
                            '№': w['id'], 'Работа': w['name'][:46],
                            'Влияние, %': round(w['influence'] * 100, 1),
                            'Конец': fmt_date(w['end']),
                        } for w in sol['late'][:12]]), hide_index=True, use_container_width=True)

                with st.expander("Как это посчитано"):
                    st.markdown(
                        f"1) Нужный множитель: **цель / план** = {fmt(sol['target'])} / {fmt(sol['plan'])} "
                        f"= **{sol['m_needed']:.4f}**\n\n"
                        f"2) Обращение калибровки (m = 1 + доверие·чувствительность·(m_сырое − 1)) даёт "
                        f"требуемую ценность показателя: **{sol['V_old']:.4f} → {sol['V_target']:.4f}**\n\n"
                        f"3) Ценность монотонно растёт с деньгами (логарифм), поэтому нужный бюджет найден "
                        f"бинарным поиском: масштаб **×{sol['scale']:.3f}** к текущим деньгам работ.\n\n"
                        f"4) Достигнутая ценность: **{sol['V_new']:.4f}** → прогноз квартала "
                        f"**{fmt(sol['forecast_after'])}**.")

                if sol.get('side_effects'):
                    with st.expander("Побочные эффекты на другие показатели"):
                        st.caption("Те же работы влияют и на другие KPI — вот их ценность после изменения.")
                        st.dataframe(pd.DataFrame([{
                            'Показатель': str(engine.G.nodes[s['kpi']].get('name', s['kpi']))[:50],
                            'Ценность после': s['value_after'],
                        } for s in sol['side_effects']]), hide_index=True, use_container_width=True)

                if st.button("✅ Применить этот бюджет к модели", use_container_width=True, key="rev_apply"):
                    if engine.apply_target_solution(sol):
                        # Как и прямой сценарий («Утвердить»), обратное решение ФИКСИРУЕМ в
                        # базовом плане на диск: без save_baseline деньги работ менялись только
                        # в памяти и терялись при первой же пересборке движка.
                        ps.save_baseline(ctx, engine.export_state())
                        st.session_state['rev_sol'] = None
                        # Деньги нескольких работ изменились — сбрасываем кеши таблиц финансов
                        # и текущего сценария, чтобы микшер/план-график перечитали новые значения.
                        for _k in [k for k in st.session_state.keys() if str(k).startswith('finance_profile_')]:
                            st.session_state.pop(_k, None)
                        st.session_state['simulation_results'] = None
                        st.session_state['scenario_params'] = None
                        st.session_state['ai_reports'] = {}
                        st.session_state['table_nonce'] = st.session_state.get('table_nonce', 0) + 1
                        st.success("Бюджет применён и сохранён в базовый план. План-график и прогнозы пересчитаны.")
                        st.rerun()
                    else:
                        st.error("Не удалось применить решение.")

    # ----------------------------------------------------------------------
    # ВКЛАДКА 3 — ЧУВСТВИТЕЛЬНОСТЬ И ОПТИМИЗАЦИЯ БЮДЖЕТА
    # ----------------------------------------------------------------------
    with tab_analytics:
        st.session_state.setdefault('sensitivity', None)
        st.session_state.setdefault('realloc', None)

        a_left, a_right = st.columns([1.25, 1])

        with a_left:
            section("Рычаги влияния", "где деньги и сроки работают сильнее всего")
            st.caption("Рычаг — прирост KPI на единицу бюджета; эластичность сравнима между работами.")
            with st.expander("Как читать эти метрики"):
                st.markdown(
                    "- **Рычаг (ΔKPI/ед.)** — на сколько вырастет KPI, если добавить одну единицу бюджета. "
                    "Чем выше — тем выгоднее вложение именно сюда.\n"
                    "- **Эластичность** — процент прироста KPI на процент прироста бюджета; безразмерна, "
                    "поэтому работы разного масштаба можно сравнивать напрямую.\n"
                    "- **ΔKPI / +срок** — как изменится KPI при продлении срока работы. Часто отрицательна: "
                    "затягивание сроков снижает ценность."
                )
            cc1, cc2, cc3 = st.columns([1, 1, 1.1])
            bump = cc1.slider("Шаг бюджета, %", 1, 25, 5, help="Размер пробного приращения для оценки производной.")
            days = cc2.slider("Шаг срока, дней", 7, 90, 30)
            run_sens = cc3.button("🔬 Рассчитать рычаги", type="primary", use_container_width=True)

            if run_sens:
                with st.spinner("Оцениваю предельную отдачу каждой работы…"):
                    st.session_state.sensitivity = engine.sensitivity_analysis(
                        budget_bump_pct=bump / 100.0, days_bump=days
                    )

            sens = st.session_state.sensitivity
            if sens:
                kpi_name_by_id = {k: engine.G.nodes[k].get('name', k) for k in kpi_ids}
                opts = ["Все KPI"] + [kpi_name_by_id[k] for k in kpi_ids]
                pick = st.selectbox("Показатель", opts, index=0)
                rows = sens if pick == "Все KPI" else [r for r in sens if r['kpi_name'] == pick]

                if rows:
                    df = pd.DataFrame([{
                        "Работа": r['leaf_name'],
                        "KPI": r['kpi_name'],
                        "Бюджет": r['F'],
                        "Рычаг (ΔKPI/ед.)": r['leverage'],
                        "Эластичность": r['elasticity_F'],
                        "ΔKPI / +срок": r['dKPI_per_month'],
                    } for r in rows])
                    lev_max = max((r['leverage'] for r in rows), default=0.0) or 1.0

                    def lev_shade(v):
                        frac = max(0.0, min(1.0, safe_float(v) / lev_max))
                        return f'background-color: rgba(79,70,229,{0.06 + 0.34*frac:.3f}); font-weight:600;'

                    st.dataframe(
                        df.style.format({
                            "Бюджет": lambda v: fmt(v),
                            "Рычаг (ΔKPI/ед.)": lambda v: fmt_sci(v),
                            "Эластичность": lambda v: fmt_sci(v),
                            "ΔKPI / +срок": lambda v: (("+" if v >= 0 else "") + fmt_sci(v)),
                        }).map(lev_shade, subset=["Рычаг (ΔKPI/ед.)"]),
                        width="stretch", hide_index=True, height=320,
                    )
                    top = df.sort_values("Рычаг (ΔKPI/ед.)", ascending=False).head(8)
                    chart = top.set_index("Работа")["Рычаг (ΔKPI/ед.)"]
                    st.caption("Топ работ по рычагу влияния")
                    st.bar_chart(chart, color="#4F46E5")
                else:
                    st.info("Для выбранного KPI нет питающих работ.")
            else:
                st.info("Нажмите «Рассчитать рычаги», чтобы увидеть, какие работы сильнее всего влияют на KPI.")

        with a_right:
            section("Оптимизация бюджета", "куда вложить дополнительные средства")
            st.caption("Жадное распределение с учётом убывающей отдачи: каждая порция уходит туда, "
                       "где даёт наибольший прирост KPI здесь и сейчас.")
            if not kpi_ids:
                st.info("Нет KPI для оптимизации.")
            else:
                with st.container(border=True):
                    kn = {engine.G.nodes[k].get('name', k): k for k in kpi_ids}
                    target_name = st.selectbox("Целевой KPI", list(kn.keys()))
                    target_id = kn[target_name]
                    pool = st.number_input(f"Дополнительный бюджет, {mc.budget_unit(engine.config.budget_scale)}",
                                           min_value=0.0, value=50.0, step=10.0)
                    steps = st.slider("Дробность распределения", 4, 24, 10,
                                      help="На сколько порций делить бюджет при жадном поиске.")
                    run_opt = st.button("🧭 Подобрать распределение", type="primary", use_container_width=True)

                if run_opt:
                    with st.spinner("Ищу распределение с максимальной отдачей…"):
                        st.session_state.realloc = engine.suggest_reallocation(
                            target_id, pool=float(pool), steps=int(steps)
                        )

                rec = st.session_state.realloc
                if rec is not None:
                    before, after = rec['kpi_before'], rec['kpi_after']
                    gain = (after - before) / before if before > 0 else 0.0
                    st.markdown(
                        f'<div class="mx-card"><div class="lbl">Ценность KPI</div>'
                        f'<div class="row"><span class="big">{fmt(after)}</span>{pct_badge(gain)}</div>'
                        f'<div class="dlt">было: {fmt(before)}</div>'
                        f'{meter_html(before, after)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    if rec['allocations']:
                        items = sorted(rec['allocations'].items(), key=lambda x: -x[1])
                        deltas = rec.get('deltas', {})
                        adf = pd.DataFrame([{
                            "№": l,
                            "Работа": rec['names'][l],
                            "Добавить бюджет": amt,
                            "Δ KPI": deltas.get(l, 0.0),
                        } for l, amt in items])
                        st.dataframe(
                            adf.style.format({"Добавить бюджет": lambda v: "+" + fmt(v),
                                              "Δ KPI": lambda v: (("+" if v >= 0 else "") + fmt_sci(v))}),
                            width="stretch", hide_index=True,
                            column_config={"№": st.column_config.TextColumn("№", width="small")},
                        )
                        alloc_chart = pd.DataFrame(
                            {"Бюджет": [amt for _, amt in items]},
                            index=[f"{l} — {rec['names'][l]}" for l, _ in items],
                        )
                        st.bar_chart(alloc_chart, color="#157347", height=160)
                        st.caption("«Δ KPI» — вклад работы в прирост показателя. Суммы — рекомендация; "
                                   "примените их вручную через микшер задач и утвердите сценарий.")
                    else:
                        st.info("При текущих параметрах дополнительный бюджет не даёт ощутимого прироста KPI.")

        if engine.schedule_violations:
            st.divider()
            section("Календарные нестыковки", "окна подзадач выходят за окно родителя")
            vrows = []
            for parent, items in engine.schedule_violations.items():
                p_name = engine.G.nodes[parent].get('name', parent)
                for it in items:
                    vrows.append({
                        "Родитель": f"{parent} — {p_name}",
                        "Подзадача": f"{it['child']} — {it['child_name']}",
                        "Раньше начала, дн.": it['start_before_days'],
                        "Позже окончания, дн.": it['end_after_days'],
                    })
            st.dataframe(pd.DataFrame(vrows), width="stretch", hide_index=True)
            st.caption("Окно подзадачи должно лежать внутри окна родителя. Скорректируйте сроки в микшере.")

        section("Из чего складывается KPI (атрибуция по Шепли)", "корректное разложение с суммой 100%")
        if kpi_ids:
            sk = st.selectbox("Показатель", options=kpi_ids,
                              format_func=lambda k: engine.G.nodes[k].get('name', k), key="shapley_kpi")
            st.caption("Метод Шепли честно делит ценность KPI между работами с учётом их взаимодействий — "
                       "в отличие от «влияния», вклады складываются в полное значение показателя. Расчёт "
                       "приближённый (Монте-Карло) и охватывает ВСЕ питающие работы.")
            all_works = st.checkbox("Считать все работы по отдельности", value=True, key="shapley_all",
                                    help="Снимите галочку, чтобы для очень крупных деревьев показать топ-12 + «прочие» (быстрее).")
            if st.button("Рассчитать вклад работ", key="run_shapley"):
                n_leaves = len([d for d in nx.ancestors(engine.G, sk)
                                if engine.G.in_degree(d) == 0 and str(engine.G.nodes[d].get('type','')).upper() != 'KPI'])
                smp = min(600, max(120, n_leaves * 40))
                with st.spinner(f"Считаю вклады по {n_leaves} работам (Монте-Карло, {smp} выборок)…"):
                    rows = engine.shapley_attribution(sk, max_players=(None if all_works else 12), samples=smp)
                if rows:
                    df = pd.DataFrame([{'Работа': r['Работа'], 'Вклад': round(r['Вклад'], 3),
                                        'Доля, %': round(r['Доля'] * 100, 1)} for r in rows])
                    st.dataframe(df, width="stretch", hide_index=True)
                    st.caption(f"Работ учтено: {len(rows)} · сумма долей = {sum(r['Доля'] for r in rows)*100:.0f}%.")
                else:
                    st.info("Для этого KPI нет питающих работ.")

        section("Граф дерева целей", "красные — проседающие узлы; толщина связи — вес влияния")
        n_nodes = engine.G.number_of_nodes()
        if n_nodes == 0:
            st.info("Граф пуст.")
        elif n_nodes > 400:
            st.warning(f"В графе {n_nodes} узлов — для читаемости и скорости интерактивный граф "
                       "отключён. Используйте таблицу план-графика и фильтры.")
        else:
            try:
                import streamlit.components.v1 as _components
                _components.html(build_goal_graph_html(engine, height_px=580, dark=theme_is_dark()), height=600, scrolling=False)
                st.caption("◼ KPI · ● работа (синий — листовая, серый — составная) · "
                           "🔴 красный — завершение позже плана (проседание). Колесо/перетаскивание — "
                           "масштаб и навигация. Граф загружает библиотеку визуализации из интернета.")
            except Exception as _e:
                st.info("Не удалось отобразить граф в этой среде.")

    # ----------------------------------------------------------------------
    # ВКЛАДКА 3 — ЭКСПЕРТНАЯ НАСТРОЙКА
    # ----------------------------------------------------------------------
    with tab_expert:
        section("Влияние и веса связей по каждому KPI", "вес — настройка; влияние — измеряемый результат")
        st.caption("Единое понятие — **влияние на KPI** (насколько просядет показатель, если убрать вклад "
                   "работы; это же — «влияние» (доля участия) и основа рычага/эластичности). **Вес** и **Тип** — "
                   "это НАСТРАИВАЕМЫЙ ВХОД, который формирует влияние. Меняете вес/тип → меняется влияние.")
        with st.expander("Вес ↔ Влияние: в чём разница"):
            st.markdown(
                "- **Влияние на KPI** (выход, измеряется): доля, на которую падает значение KPI при "
                "обнулении вклада работы. На уровне KPI это и есть *share*; рычаг и эластичность — то же "
                "влияние на единицу/в % бюджета.\n"
                "- **Вес связи** (вход, настраивается): сила связи ребёнок→родитель, нормируется среди "
                "соседей; вместе с типом и нелинейной агрегацией ПОРОЖДАЕТ влияние.\n"
                "- Поэтому их не сводят в одно число: вес — это «ручка», влияние — «результат». "
                "Колонка «Влияние на KPI» показывает результат текущих весов/типов."
            )
        with st.expander("Что означают типы зависимости"):
            st.markdown(
                "- **Линейный** — вклад пропорционален ценности (`w·v`).\n"
                "- **Насыщающий** — убывающая отдача: большой вклад добавляет всё меньше (`w·ln(1+v)`).\n"
                "- **Пороговый** — учитывается только выше «критической массы» (`w·v·σ(k·(v−τ))`).\n"
                "- **Усиливающий** — синергия: сильный вклад весомее, рост до ×2 (`w·v·(1+tanh(v/τ))`).\n"
                "- **Тормозящий** — снижает значение показателя (`−w·v`): риск или конкуренция за ресурс."
            )

        weight_rows = engine.iter_weight_rows()
        type_labels = list(RELATION_LABELS.values())
        label_to_key = {v: k for k, v in RELATION_LABELS.items()}
        if not weight_rows:
            st.info("Матрица весов пуста — нет многодетных связей для настройки.")
        else:
            kpi_order = []
            for r in weight_rows:
                if (r['kpi_id'], r['KPI']) not in kpi_order:
                    kpi_order.append((r['kpi_id'], r['KPI']))
            kpi_names = [name for _, name in kpi_order]
            name_to_id = {name: kid for kid, name in kpi_order}
            sel_name = st.selectbox("Показатель (KPI)", kpi_names, key="weights_kpi_pick",
                                    help="Показать веса и влияние связей только этого показателя.")
            sel_kid = name_to_id.get(sel_name)

            sub = [r for r in weight_rows if r['kpi_id'] == sel_kid]
            data = [{
                'Источник': r['Источник'],
                'Родитель': r['Родитель'],
                'Вес': round(float(r['Вес']), 4),
                'Влияние': (round(float(r['Влияние']) * 100.0, 1) if r['Влияние'] is not None else None),
                'Тип': r['Тип'],
            } for r in sub]
            pdf = pd.DataFrame(data)

            edited = st.data_editor(
                pdf, width="stretch", hide_index=True, num_rows="fixed",
                column_config={
                    'Источник': st.column_config.TextColumn(disabled=True, width="large"),
                    'Родитель': st.column_config.TextColumn(disabled=True, width="large"),
                    'Вес': st.column_config.NumberColumn("Вес (вход)", min_value=0.0, max_value=1.0,
                                                         step=0.01, format="%.4f",
                                                         help="Настраиваемая сила связи ребёнок→родитель."),
                    'Влияние': st.column_config.NumberColumn("Влияние на KPI (выход)", disabled=True,
                                                             format="%.1f%%",
                                                             help="Доля просадки этого KPI при обнулении работы."),
                    'Тип': st.column_config.SelectboxColumn("Тип связи", options=type_labels, width="small",
                                                            help="Форма влияния вклада ребёнка на родителя."),
                },
                key=f"weights_editor_{sel_kid}",
            )
            st.caption(f"Показатель: «{sel_name}» · связей: {len(sub)}. "
                       "«Вес» — вход (редактируется), «Влияние» — измеряемый выход (%). "
                       "Чтобы увидеть влияние тех же связей на другой KPI — переключите показатель выше.")
            if st.button("Применить веса и пересчитать", type="primary"):
                for r, new_w, new_t in zip(sub, edited['Вес'].tolist(), edited['Тип'].tolist()):
                    rt = label_to_key.get(str(new_t), 'linear')
                    engine.set_weight(r['kpi_id'], r['source_id'], r['target_id'],
                                      safe_float(new_w), relation_type=rt)
                engine.recalculate()
                st.session_state.simulation_results = None
                st.session_state.ai_reports = {}
                st.success("Веса и типы обновлены, базовая модель пересчитана.")
                st.rerun()

        section("Калибровка точности по KPI", "повышение достоверности прогноза без истории")
        st.caption("Без исторических данных прогноз заземляют на методику и экспертное суждение. "
                   "**Чувствительность** — насколько сильно ресурсы двигают KPI (1 — как в модели, "
                   "<1 — слабее, много внешних факторов; >1 — сильнее). **Уверенность** <1 «притягивает» "
                   "прогноз к плану там, где доверия меньше (снижает ожидаемую ошибку). **Связь** — "
                   "прямой (переменные формулы производит план) или косвенный (план влияет лишь "
                   "опосредованно, напр. ремонт → привлекательность → балл; уверенность ниже).")
        calib_rows = engine.iter_calibration_rows()
        if not calib_rows:
            st.info("Нет показателей для калибровки.")
        else:
            indirect = [r for r in calib_rows if str(r.get('Связь')) == 'косвенный']
            if indirect:
                st.warning(f"Косвенно связаны с планом: {len(indirect)} из {len(calib_rows)} KPI — "
                           "их прогноз является сценарием‑допущением, а не предсказанием.")
            with st.expander("Формулы KPI и связь с планом (из методики)"):
                for r in calib_rows:
                    link = r.get('Связь', '—')
                    badge = "🟢 прямой" if link == 'прямой' else ("🟠 косвенный" if link == 'косвенный' else "⚪ —")
                    st.markdown(f"**{r['KPI']}** — {badge}")
                    if r.get('Формула'):
                        st.caption(f"формула: {r['Формула']}")
                    if r.get('Обоснование'):
                        st.caption(f"связь с планом: {r['Обоснование']}")
            cdf = pd.DataFrame(calib_rows)
            cview = cdf[['KPI', 'Тип KPI', 'Связь', 'Чувствительность', 'Уверенность', 'Драйвер', 'Потолок', 'Режим', 'ρ (CES)']]
            cedit = st.data_editor(
                cview, width="stretch", hide_index=True, num_rows="fixed",
                column_config={
                    "KPI": st.column_config.TextColumn(disabled=True),
                    "Тип KPI": st.column_config.TextColumn(disabled=True, width="small",
                                                           help="Определён по топологии: показатель=задача / несколько задач / общий показатель."),
                    "Связь": st.column_config.TextColumn("Связь плана с KPI", disabled=True, width="small",
                                                         help="прямой / косвенный (как переменные формулы связаны с планом)."),
                    "Чувствительность": st.column_config.NumberColumn(min_value=0.2, max_value=2.0, step=0.05, format="%.2f",
                                                                       help="Масштаб реакции KPI на ресурсы (1 — нейтрально)."),
                    "Уверенность": st.column_config.NumberColumn(min_value=0.3, max_value=1.0, step=0.05, format="%.2f",
                                                                  help="<1 — прогноз ближе к плану (меньше доверия модели)."),
                    "Драйвер": st.column_config.SelectboxColumn(options=["бюджет", "срок", "оба"], required=True, width="small"),
                    "Потолок": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.0f",
                                                             help="Максимум значения KPI (100 для долей/процентов). Пусто — без потолка."),
                    "Режим": st.column_config.SelectboxColumn(options=["классический", "ces"], required=True, width="small",
                                                              help="ces — единый CES-агрегатор вместо типов связей."),
                    "ρ (CES)": st.column_config.NumberColumn(min_value=-8.0, max_value=4.0, step=0.5, format="%.1f",
                                                             help="ρ=1 — сумма (заменители); ρ→0 — геометрическое; ρ<0 — «слабое звено» (комплементарность)."),
                },
                key="calib_editor",
            )
            st.caption("Режим **ces** включает CES-агрегатор: одна ручка ρ задаёт форму свёртки вкладов "
                       "(сумма ↔ геометрическое ↔ минимум), заменяя «зоопарк» типов связей для этого KPI.")
            if st.button("Применить калибровку", type="primary", key="apply_calib"):
                for orig, s_, c_, d_, vmax_, mode_, rho_ in zip(
                        calib_rows, cedit['Чувствительность'].tolist(), cedit['Уверенность'].tolist(),
                        cedit['Драйвер'].tolist(), cedit['Потолок'].tolist(),
                        cedit['Режим'].tolist(), cedit['ρ (CES)'].tolist()):
                    vm = '' if (vmax_ is None or (isinstance(vmax_, float) and pd.isna(vmax_))) else vmax_
                    engine.set_kpi_calibration(orig['kpi_id'], sensitivity=safe_float(s_),
                                               confidence=safe_float(c_), driver=str(d_), value_max=vm,
                                               agg_mode=str(mode_), ces_rho=safe_float(rho_))
                engine.recalculate()
                st.session_state.simulation_results = None
                st.session_state.ai_reports = {}
                st.success("Калибровка обновлена. Прогнозы пересчитаны.")
                st.rerun()

        section("Зависимости предшествования", "срыв срока каскадом двигает зависимые работы")
        st.caption("Связь «А → Б» (finish-to-start): Б не может начаться раньше окончания А. При утверждении "
                   "переноса срока предшественника зависимые работы автоматически сдвигаются по цепочке "
                   "(длительность сохраняется).")
        nodes_all = [n for n in engine.G.nodes if str(engine.G.nodes[n].get('type', '')).upper() != 'KPI']
        nodes_all.sort()
        flbl = lambda n: f"{n} — {engine.G.nodes[n].get('name', n)}"
        c1, c2, c3 = st.columns([3, 3, 1])
        pred = c1.selectbox("Предшественник (А)", options=nodes_all, format_func=flbl, key="prec_pred")
        succ = c2.selectbox("Последователь (Б)", options=nodes_all, format_func=flbl, key="prec_succ")
        if c3.button("Добавить", use_container_width=True):
            engine.set_precedence(pred, succ, on=True)
            st.success(f"Связь {pred} → {succ} добавлена.")
            st.rerun()
        edges_p = engine.get_precedence_edges()
        if edges_p:
            for p, s in edges_p:
                cc1, cc2 = st.columns([6, 1])
                cc1.markdown(f"**{flbl(p)}** → **{flbl(s)}**")
                if cc2.button("Удалить", key=f"rmprec_{p}_{s}", use_container_width=True):
                    engine.set_precedence(p, s, on=False)
                    st.rerun()
        else:
            st.caption("Зависимостей пока нет.")

    with tab_unmatched:
        section("Несвязанные финансы", "сущности из файла финансов, которые не удалось привязать к план-графику")
        if not hasattr(engine, 'unmatched_finances') or not engine.unmatched_finances:
            st.success("Все финансовые строки успешно связаны с план-графиком или файл финансов не загружен.")
        else:
            st.warning(f"Найдено {len(engine.unmatched_finances)} записей в файле финансов, которых нет в план-графике (не совпал «№ п/п»).")
            
            rows = []
            for item in engine.unmatched_finances:
                row = {'№ п/п': item['id'], 'Наименование': item['name']}
                for y, prof in item['profile'].items():
                    row[f'{y} База'] = prof.get('base', 0.0)
                    row[f'{y} Потребн. (сверх)'] = prof.get('req_extra', 0.0)
                    row[f'{y} Доп.'] = prof.get('add', 0.0)
                rows.append(row)
                
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Проверьте номера пунктов (ID) в исходных Excel-файлах: они должны совпадать.")

def page_portfolio():
    apply_theme()
    st.markdown("# 🎯 Портфель проектов")
    st.markdown('<div class="mx-sub">Межпроектное распределение дополнительного бюджета: куда вложить, '
                'чтобы корзина KPI выросла сильнее всего (с учётом убывающей отдачи).</div>',
                unsafe_allow_html=True)
    mgr = get_build_manager()
    ready = {}
    for c in ps.discover(ps.PROJECTS_ROOT):
        stt = mgr.get(c.slug)
        if stt and stt.get('status') == 'ready' and stt.get('engine') is not None:
            ready[c.slug] = stt['engine']
    if not ready:
        st.info("Нет загруженных проектов. Откройте проекты во вкладке «Микшер», чтобы они собрались, "
                "затем вернитесь сюда. Уже собранные проекты участвуют в распределении.")
        return
    st.caption(f"Готовых проектов в портфеле: {len(ready)} — {', '.join(ready.keys())}")
    c1, c2 = st.columns(2)
    extra = c1.number_input("Дополнительный бюджет, млн ₽", min_value=0.0, value=100.0, step=10.0)
    step = c2.number_input("Шаг распределения, млн ₽", min_value=1.0, value=10.0, step=1.0)
    if st.button("🧭 Распределить по портфелю", type="primary"):
        with st.spinner("Жадный подбор распределения…"):
            res = mc.portfolio_greedy_allocation(ready, float(extra), float(step))
        if not res['allocations']:
            st.warning("Положительной отдачи не найдено — возможно, бюджет/шаг слишком малы.")
        else:
            st.markdown(f"**Распределено: {res['spent']} млн ₽**")
            st.dataframe(pd.DataFrame([
                {'Проект': a['project'], 'Работа': f"{a['node']} — {a['name']}",
                 'Добавлено, млн': a['budget_added']} for a in res['allocations']
            ]), width="stretch", hide_index=True)
            st.markdown("**Прирост корзины KPI по проектам:**")
            st.dataframe(pd.DataFrame([
                {'Проект': s, 'Было': g['было'], 'Стало': g['стало'], 'Прирост, %': g['прирост_%']}
                for s, g in res['project_gain'].items()
            ]), width="stretch", hide_index=True)
            st.caption("Распределение неразрушающее (предложение). Чтобы применить — внесите бюджеты в "
                       "соответствующих проектах во вкладке «Микшер».")


# ======================================================================
# МАРШРУТИЗАЦИЯ (мультистраничный режим)
# ======================================================================
ps.migrate_legacy()  # однократный перенос data/ → projects/<...>
_projects = ps.discover(ps.PROJECTS_ROOT)
_active = st.session_state.get('active_project')
if _active and not any(c.slug == _active for c in _projects):
    _active = None
    st.session_state.active_project = None


_pages = [st.Page(page_projects, title="Проекты", icon="📁", url_path="projects",
                  default=(_active is None))]
if _active:
    _pages.append(st.Page(page_mixer, title="Микшер", icon="🎛️", url_path="mixer", default=True))
_pages.append(st.Page(page_portfolio, title="Портфель", icon="🎯", url_path="portfolio"))
st.navigation(_pages).run()