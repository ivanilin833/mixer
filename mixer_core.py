# mixer_core.py

import random
import copy
import os
import logging
import json
import re
import hashlib
import time
import threading
import networkx as nx
import pandas as pd
import numpy as np
import pypandoc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI


class _LLMGate:
    """Глобальная ПОСЛЕДОВАТЕЛЬНАЯ очередь к локальной LLM (Ollama считает по сути по одному).

    Сериализует фактические запросы из всех проектов/потоков: вместо того чтобы параллельные
    сборки толкались и ловили таймауты, запросы выполняются по одному. `waiting` показывает,
    сколько вызовов сейчас ждут+выполняются — для статуса «в очереди (N перед вами)»."""
    def __init__(self):
        self._run_lock = threading.Lock()   # пропускает ровно один вызов за раз
        self._count_lock = threading.Lock()
        self._waiting = 0

    @property
    def waiting(self) -> int:
        with self._count_lock:
            return self._waiting

    def __enter__(self):
        with self._count_lock:
            self._waiting += 1
        self._run_lock.acquire()
        return self

    def __exit__(self, *exc):
        self._run_lock.release()
        with self._count_lock:
            self._waiting -= 1
        return False


_LLM_GATE = _LLMGate()


def llm_queue_depth() -> int:
    """Сколько запросов к LLM сейчас в очереди (ждут + выполняется). 0 — модель свободна."""
    return _LLM_GATE.waiting


def _atomic_write_text(path: str, text: str):
    """Атомарная запись текстового файла (temp + rename) — без рваных файлов при сбое/гонке."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def clean_float(value) -> float:
    """Преобразует строку с пробелами/запятой/неразрывным пробелом в float (#12).

    Делегирует в parse_ru_number (определён ниже): понимает «1 234,56», неразрывные
    и узкие пробелы, валюту; пустое/нечитаемое → 0.0 (не None)."""
    return parse_ru_number(value, 0.0)

# ==========================================
# 1. КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ
# ==========================================
# Логирование с разделением ПО ПРОЕКТАМ:
#  • каждый проект пишет в свой файл (все его сообщения);
#  • общий лог (mixer.log) получает СИСТЕМНЫЕ сообщения (без привязки к проекту)
#    и ДУБЛИ всех ошибок (ERROR и выше) из любого проекта;
#  • терминал получает всё.
# Текущий проект хранится в contextvars — это корректно работает и в фоновых потоках
# (каждый поток сборки проекта выставляет свой контекст и пишет в свой лог).
import contextvars

_current_project = contextvars.ContextVar("mixer_project", default=None)
_LOG_FMT = logging.Formatter('%(asctime)s [%(levelname)s]%(proj)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class _ContextFilter(logging.Filter):
    """Проставляет в запись текущий проект (для маршрутизации по файлам и в формат)."""
    def filter(self, record):
        slug = _current_project.get()
        record.project = slug
        record.proj = f" [{slug}]" if slug else ""
        return True


class _SystemFilter(logging.Filter):
    """Общий лог: системные записи (без проекта) ИЛИ любые ошибки (дублируем)."""
    def filter(self, record):
        return getattr(record, 'project', None) is None or record.levelno >= logging.ERROR


class _ProjectFilter(logging.Filter):
    """Лог конкретного проекта: только записи с этим slug."""
    def __init__(self, slug):
        super().__init__()
        self.slug = slug
    def filter(self, record):
        return getattr(record, 'project', None) == self.slug


logger = logging.getLogger("ProjectMixer")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addFilter(_ContextFilter())  # фильтр на логгере выполняется для всех записей

if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in logger.handlers):
    _sh = logging.StreamHandler()
    _sh.setFormatter(_LOG_FMT)
    logger.addHandler(_sh)  # терминал — всё

if not any(getattr(h, "_mixer_common", False) for h in logger.handlers):
    try:
        _log_path = os.environ.get("MIXER_LOG_FILE", "mixer.log")
        _fh = logging.FileHandler(_log_path, encoding='utf-8')
        _fh.setFormatter(_LOG_FMT)
        _fh.addFilter(_SystemFilter())   # общий лог — система + дубли ошибок
        _fh._mixer_common = True
        logger.addHandler(_fh)
    except Exception as _e:
        logger.warning(f"Не удалось включить общий лог: {_e}")

_project_log_handlers: Dict[str, logging.Handler] = {}


def register_project_log(slug: str, log_path: str):
    """Подключает (один раз) файловый лог для проекта: туда попадут все его сообщения."""
    if not slug or slug in _project_log_handlers:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        h = logging.FileHandler(log_path, encoding='utf-8')
        h.setFormatter(_LOG_FMT)
        h.addFilter(_ProjectFilter(slug))
        logger.addHandler(h)
        _project_log_handlers[slug] = h
    except Exception as e:
        logger.warning(f"Не удалось включить лог проекта {slug}: {e}")


def unregister_project_log(slug: str):
    """Отключает и ЗАКРЫВАЕТ файловый лог проекта (освобождает дескриптор/память).
    Вызывать при удалении проекта, чтобы хендлеры не накапливались в процессе."""
    h = _project_log_handlers.pop(slug, None)
    if h is not None:
        try:
            logger.removeHandler(h)
            h.close()
        except Exception:
            pass


def set_log_project(slug: Optional[str], log_path: Optional[str] = None):
    """Устанавливает текущий проект для логирования (в этом потоке/контексте).
    Если задан путь — подключает файловый лог проекта. Вызывать в начале работы со
    страницей проекта и в начале фоновой сборки проекта."""
    if slug and log_path:
        register_project_log(slug, log_path)
    _current_project.set(slug)


def parse_ru_number(value, default: float = 0.0) -> float:
    """Надёжный разбор числа (#12).

    Понимает русский формат: запятая как десятичный разделитель и неразрывный/обычный
    пробел как разделитель тысяч («1 234,56»), знаки валюты и пр. Пустые значения и
    нечитаемое → default (по умолчанию 0, а НЕ None)."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            return default if (isinstance(value, float) and np.isnan(value)) else float(value)
        except (TypeError, ValueError):
            return default
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "—", "-", "n/a"):
        return default
    # убираем все виды пробелов (вкл. неразрывный \u00A0 и узкий \u202F) и валюту
    for ch in ("\u00A0", "\u202F", " ", "\t", "₽", "руб.", "руб", "р."):
        s = s.replace(ch, "")
    s = s.replace(",", ".")
    # оставляем только цифры, точку и знак минус
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", ".", "-."):
        return default
    # если точек несколько (разделители тысяч точками) — оставляем последнюю как десятичную
    if s.count(".") > 1:
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + "." + tail
    try:
        return float(s)
    except ValueError:
        return default


def stable_sigmoid(x: float) -> float:
    """Численно устойчивая логистическая функция 1/(1+e^-x) без переполнения exp (#10)."""
    x = float(np.clip(x, -60.0, 60.0))
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


# Типы зависимости ребра «ребёнок→родитель» и их человекочитаемые названия.
# Математика каждого типа реализована в ProjectMixer._edge_contribution().
RELATION_LABELS = {
    'linear':     'Линейный',      # w·v — пропорциональный вклад
    'saturating': 'Насыщающий',    # w·ln(1+v)/ln2 — убывающая отдача
    'threshold':  'Пороговый',     # w·v·σ(k·(v−τ)) — вклад только выше критической массы
    'amplifying': 'Усиливающий',   # w·v·(1+tanh(v/τ)) — синергия, до ×2
    'inhibitory': 'Тормозящий',    # −w·v — снижает значение KPI
}
# Провенансные метки от LLM/построения графа считаются линейной математикой.
_LINEAR_ALIASES = {'linear', 'direct', 'semantic', 'equal', 'manual', ''}


def canonical_relation(rt) -> str:
    """Приводит тип к канону для математики: всё, кроме спец-типов, → 'linear'.

    Понимает и английские ключи, и русские названия из RELATION_LABELS (как может вернуть LLM)."""
    s = str(rt or 'linear').strip().lower()
    if s in RELATION_LABELS:
        return s
    # обратное соответствие по русским названиям
    for key, label in RELATION_LABELS.items():
        if s == label.lower():
            return key
    return 'linear'

# Единицы измерения бюджета: подпись для интерфейса + множитель приведения к МИЛЛИОНАМ.
# factor = во сколько раз единица меньше миллиона (млн×1, сотни тыс×0.1, тыс×0.001, руб×1e-6).
BUDGET_UNITS = {
    'millions':       {'short': 'млн ₽',       'full': 'млн руб.',         'factor': 1.0},
    'hundred_thousands': {'short': 'сот. тыс ₽', 'full': 'сотни тыс. руб.', 'factor': 0.1},
    'thousands':      {'short': 'тыс ₽',       'full': 'тыс. руб.',        'factor': 1e-3},
    'rub':            {'short': '₽',           'full': 'руб.',             'factor': 1e-6},
}

def budget_unit(scale: str, key: str = 'short') -> str:
    """Подпись единицы бюджета для интерфейса ('млн ₽' / 'тыс ₽' / '₽')."""
    return BUDGET_UNITS.get(str(scale), BUDGET_UNITS['millions']).get(key, 'млн ₽')


@dataclass
class MixerConfig:
    """Коэффициенты модели ценности. Дефолты подобраны так, чтобы (1) бюджет и срок были
    СОПОСТАВИМЫ по влиянию, причём бюджет — главный рычаг «микшера»; (2) коэффициент срока
    был БЕЗРАЗМЕРНЫМ (нечувствителен к тому, в днях/неделях заданы длительности); (3) не было
    скачков от изменения срока на 1 день и переполнений.

    Локальная ценность листа:
        V_local = alpha·ln(1 + lambda_f·F) + beta·σ( -sigmoid_k·(Δ/T_opt − 1) )
      • alpha·ln(1+lambda_f·F) — бюджетный член с убывающей отдачей; «колено» при F≈1/lambda_f.
        При lambda_f=0.01 колено ≈ 100 (млн ₽): около типичных бюджетов бюджет влияет заметно.
      • beta·σ(...) — временной член ОТ ОТНОСИТЕЛЬНОГО отклонения срока Δ/T_opt (а не от разницы
        в днях). На сроке = плановому → 0.5·beta; вдвое дольше → ~0.12·beta; быстрее → до beta.

    Агрегация снизу вверх:
        agg = local + raw_sum·σ( sigmoid_activation_k·(raw_sum − activation_threshold) )
      Мягкий «порог»: вклад ниже activation_threshold подавляется (шумовой пол), выше — почти
      линеен. Монотонен по raw_sum (нет инверсий)."""
    alpha: float = 2.0                    # вес бюджета
    beta: float = 1.5                     # вес срока (вторичный рычаг)
    lambda_f: float = 0.01                # масштаб бюджета: «колено» отдачи при F≈100
    # Единица измерения бюджета в план-графике. От неё зависит эффективный масштаб lambda_f:
    # 'millions' — млн (как раньше), 'thousands' — сотни тысяч/тысячи, 'rub' — рубли.
    budget_scale: str = 'millions'
    sigmoid_k: float = 2.0                # БЕЗРАЗМЕРНАЯ чувствительность к относит. отклонению срока
    activation_threshold: float = 0.5     # шумовой пол агрегации
    sigmoid_activation_k: float = 1.0     # мягкость порога активации
    deficit_penalty_power: float = 2.0    # не используется
    time_penalty_power: float = 0.5       # авто-продление срока: √-зависимость (мягко)
    time_bonus_enabled: bool = True       # опережение графика немного повышает ценность
    # Режим прогноза: 'value' — по отношению абстрактных ценностей (текущий);
    # 'completion' — по ДОЛЕ ВЫПОЛНЕНИЯ ПЛАНА (вариант Б): KPI = взвешенная свёртка долей
    # выполнения работ (веса = влияние), 1.0 = всё по плану → прогноз = план.
    forecast_mode: str = 'value'
    # CES по умолчанию: единый агрегатор вместо порога активации и «зоопарка» типов связей.
    # Одна ручка ρ задаёт форму свёртки (ρ=1 — взвешенная сумма; <1 — комплементарность).
    default_agg_mode: str = 'ces'
    default_ces_rho: float = 1.0
    # Штраф за ПОЗДНЕЕ ЗАВЕРШЕНИЕ обычной работы (не только отклонение длительности от оптимума).
    late_finish_penalty_enabled: bool = False  # по умолчанию выключен (поведение не меняется)
    late_finish_weight: float = 0.5            # доля β, на которую наказывается сильное опоздание
    # --- НОВОЕ: Вектор А и Б (Шаг 2) ---
    discount_rate: float = 0.06                # Ставка дисконтирования (TVM), по умолчанию 6%
    base_year: int = 2026                      # Базовый год для расчета дисконта
    # Вероятности финансирования по статусам (риск-взвешивание). «База» всегда 1.0 (гарантирована).
    # «Потребность» — рискованная: по умолчанию 0.5 (учитываем наполовину, отражая неопределённость).
    # «Доп. потребность» — дефицитная/наименее вероятная: по умолчанию 0.1.
    # Значения настраиваются ползунками в микшере (стресс-тест «утвердят только базу» и т.п.).
    rho_req: float = 1.0                        # Вероятность финансирования потребности
    rho_add: float = 0.0                        # Вероятность финансирования доп. потребности

class ScheduleColumnMap:
    ID = '№ п/п'
    TYPE = 'Задача Подзадача Мероприятие Веха'
    NAME = 'Наименование задачи, результата и вехи'
    START_DATE = 'Начало реализации'
    END_DATE = 'Окончание реализации'
    BUDGET = 'Бюджет (млн. руб.)'

class IndicatorColumnMap:
    TASK_ID = 'task_id'
    NAME = 'name'
    METHODOLOGY = 'methodology'

class IndicatorMapping:
    TASK_ID = ['№', 'задач']
    NAME = ['наименование', 'показателя']
    METHODOLOGY = ['методик']
    PLAN = ['план']
    FORECAST = ['прогноз']
    FACT = ['факт']

# Распознавание кварталов в заголовках таблицы показателей.
RU_QUARTERS = {'iv квартал': 4, 'iii квартал': 3, 'ii квартал': 2, 'i квартал': 1,
               '4 квартал': 4, '3 квартал': 3, '2 квартал': 2, '1 квартал': 1}

def _col_year(col) -> Optional[int]:
    """Извлекает год (20xx) из любого уровня заголовка столбца."""
    txt = " ".join(str(c) for c in (col if isinstance(col, (tuple, list)) else [col]))
    m = re.findall(r'(20\d{2})', txt)
    return int(m[0]) if m else None

def _col_quarter(col):
    """Возвращает номер квартала 1..4 или 'Y' (годовой блок), если столбец относится к плану/прогнозу/факту."""
    txt = " ".join(str(c).lower() for c in (col if isinstance(col, (tuple, list)) else [col]))
    for key, q in RU_QUARTERS.items():
        if key in txt:
            return q
    return 'Y'

def _col_kind(col) -> Optional[str]:
    """Тип подстолбца: plan / forecast / fact (по последнему уровню заголовка)."""
    last = str(col[-1] if isinstance(col, (tuple, list)) else col).lower()
    if any(k in last for k in IndicatorMapping.PLAN): return 'plan'
    if any(k in last for k in IndicatorMapping.FORECAST): return 'forecast'
    if any(k in last for k in IndicatorMapping.FACT): return 'fact'
    return None

# ==========================================
# 2. ПАРСЕР МЕТОДИК (исходные тексты)
# ==========================================
class MethodologyParser:
    _cache = {}

    @classmethod
    def preload_methodologies(cls, file_names: List[str], base_dir: str):
        unique_files = set([f for f in file_names if pd.notna(f) and str(f).strip()])
        for file_name in unique_files:
            cls.get_text(file_name, base_dir)

    @classmethod
    def get_text(cls, file_name: str, base_dir: str) -> str:
        if not file_name or pd.isna(file_name): return ""
        file_name_str = str(file_name).strip()
        if file_name_str in cls._cache: return cls._cache[file_name_str]
        full_path = os.path.join(base_dir, file_name_str)
        content = cls._extract_full_text(full_path)
        cls._cache[file_name_str] = content
        return content

    @staticmethod
    def _extract_full_text(file_path: str) -> str:
        if not os.path.exists(file_path): return ""
        if file_path.lower().endswith('.doc'): return ""
        try:
            return pypandoc.convert_file(file_path, 'markdown', extra_args=['--wrap=none']).strip()
        except Exception as e:
            logger.error(f"Ошибка конвертации Pandoc для {file_path}: {e}")
            return ""

# ==========================================
# 3. МУЛЬТИ-ИНДЕКСНЫЙ ОРКЕСТРАТОР
# ==========================================
class DataLoaderOrchestrator:
    @staticmethod
    def safe_load_finances(filepath: str) -> dict:
        """Парсинг мультииндексного файла финансов по годам и вариантам."""
        if not filepath or not os.path.exists(filepath):
            return {'mapped': {}}
        try:
            df = pd.read_excel(filepath, header=None)
            start_idx = -1
            # Ищем строку с заголовками (где есть "№ п/п")
            for i, row in df.iterrows():
                row_str = " ".join(str(x) for x in row.values).lower()
                if '№ п/п' in row_str and 'наименование' in row_str:
                    start_idx = i
                    break
            if start_idx == -1:
                return {'mapped': {}}
                
            # Протаскиваем заголовки вправо (ffill), чтобы устранить объединенные ячейки Excel
            h0 = df.iloc[start_idx].ffill()
            h1 = df.iloc[start_idx + 1].ffill() if start_idx + 1 < len(df) else pd.Series([""] * len(h0))
            h2 = df.iloc[start_idx + 2].ffill() if start_idx + 2 < len(df) else pd.Series([""] * len(h0))

            mapped = {}
            for i in range(start_idx + 3, len(df)):
                row = df.iloc[i]
                
                # Находим ключевые столбцы (строгое приведение к str для защиты от float/NaN)
                task_id_col, name_col = -1, -1
                for c in range(len(h0)):
                    h0_c = str(h0.iloc[c]).lower()
                    h1_c = str(h1.iloc[c]).lower()
                    if '№ п/п' in h0_c or '№ п/п' in h1_c: task_id_col = c
                    if 'наименование' in h0_c or 'наименование' in h1_c: name_col = c
                
                if task_id_col == -1: break
                
                raw_id = str(row.iloc[task_id_col]).strip()
                if raw_id == 'nan' or not raw_id: continue
                if raw_id.endswith('.0'): raw_id = raw_id[:-2]
                
                task_name = str(row.iloc[name_col]).strip() if name_col != -1 else ""
                fin_profile = {}
                
                # Собираем деньги по годам
                for c in range(len(h0)):
                    # Жесткая защита: конвертируем каждый элемент в строку
                    col_l0 = str(h0.iloc[c]).lower()
                    col_l1 = str(h1.iloc[c]).lower()
                    col_l2 = str(h2.iloc[c]).lower()
                    
                    year_match = re.search(r'(20\d{2})\s*год', col_l0)
                    if year_match:
                        year = int(year_match.group(1))
                        if year not in fin_profile:
                            fin_profile[year] = {'base': 0.0, 'req_extra': 0.0, 'req_total': 0.0, 'add': 0.0}
                        
                        val = parse_ru_number(row.iloc[c], 0.0)
                        
                        if 'базовый' in col_l1:
                            fin_profile[year]['base'] = val
                        elif 'потребный' in col_l1:
                            if 'сверх базового' in col_l2:
                                fin_profile[year]['req_extra'] = val
                            elif 'всего' in col_l2:
                                fin_profile[year]['req_total'] = val
                            else:
                                fin_profile[year]['req_total'] = val
                        elif 'дополнительная' in col_l1 or 'доп' in col_l1:
                            fin_profile[year]['add'] = val

                mapped[raw_id] = {'name': task_name, 'profile': fin_profile}
            return {'mapped': mapped}
        except Exception as e:
            import traceback
            logger.error(f"Ошибка парсинга финансов: {e}\n{traceback.format_exc()}")
            return {'mapped': {}}
    
    @staticmethod
    def safe_load_excel(filepath: str, is_indicator: bool = False) -> pd.DataFrame:
        try:
            if is_indicator:
                df = pd.read_excel(filepath, header=[0, 1, 2])
                # ДОБАВЛЕНО: восстанавливаем объединённые ячейки заголовка. Год (уровень 0)
                # тянем сквозь все подстолбцы; квартал (уровень 1) — ТОЛЬКО внутри своей
                # годовой группы, иначе пустой годовой блок ошибочно получил бы «IV квартал».
                try:
                    hdr = df.columns.to_frame(index=False)
                    hdr.columns = [0, 1, 2]
                    l0 = hdr[0].astype(str)
                    l0 = l0.where(~l0.str.match(r'^Unnamed', na=False), other=pd.NA).ffill()
                    hdr[0] = l0
                    l1 = hdr[1].astype(str)
                    l1 = l1.where(~l1.str.match(r'^Unnamed', na=False), other=pd.NA)
                    hdr[1] = l1.groupby(hdr[0]).ffill()
                    df.columns = pd.MultiIndex.from_frame(hdr)
                except Exception as ex:
                    logger.warning(f"Не удалось нормализовать заголовок показателей: {ex}")
                data = []
                plan_indices = []
                task_id_idx, name_idx, methodology_idx = 0, 1, -1

                # ДОБАВЛЕНО: карта подстолбцов по (год, квартал, тип) для мультигодовой таблицы.
                # Структура из ТЗ: для каждого года — 4 квартала × (план/прогноз/факт/откл.) + годовой блок.
                period_cols = {}   # idx -> (year, q, kind)
                for idx, col in enumerate(df.columns):
                    col_text = " ".join([str(c).lower() for c in col])
                    if any(kw in col_text for kw in IndicatorMapping.TASK_ID) and '№' in str(col[-1]).lower():
                        task_id_idx = idx
                    elif any(kw in col_text for kw in IndicatorMapping.METHODOLOGY):
                        methodology_idx = idx
                    elif any(kw in col_text for kw in IndicatorMapping.NAME) and idx != methodology_idx:
                        name_idx = idx
                    if any(kw in str(col[-1]).lower() for kw in IndicatorMapping.PLAN):
                        plan_indices.append(idx)
                    yr, kind = _col_year(col), _col_kind(col)
                    if yr is not None and kind is not None and idx not in (task_id_idx, name_idx, methodology_idx):
                        period_cols[idx] = (yr, _col_quarter(col), kind)

                for _, row in df.iterrows():
                    raw_id = str(row.iloc[task_id_idx]).strip()
                    if pd.isna(row.iloc[task_id_idx]) or not raw_id or raw_id == 'nan': continue
                    if raw_id.endswith('.0'): raw_id = raw_id[:-2]
                    item = {
                        IndicatorColumnMap.TASK_ID: raw_id,
                        IndicatorColumnMap.NAME: str(row.iloc[name_idx]).strip() if pd.notna(row.iloc[name_idx]) else "",
                        IndicatorColumnMap.METHODOLOGY: str(row.iloc[methodology_idx]).strip() if methodology_idx != -1 and pd.notna(row.iloc[methodology_idx]) else ""
                    }
                    keys = ['Q1_plan', 'Q2_plan', 'Q3_plan', 'Q4_plan', 'Year_plan']
                    for i, key in enumerate(keys):
                        if len(plan_indices) > i:
                            val = row.iloc[plan_indices[i]]
                            try: item[key] = float(val) if pd.notna(val) else 0.0
                            except (ValueError, TypeError): item[key] = 0.0

                    # ДОБАВЛЕНО: собираем мультигодовые квартальные периоды (план/прогноз/факт)
                    # и ОТДЕЛЬНО — годовой блок (#1: год НЕ равен сумме кварталов).
                    periods = {}    # (year, q) -> {'plan','forecast','fact'}
                    annual_map = {}  # year -> {'plan','forecast','fact'}
                    for idx, (yr, q, kind) in period_cols.items():
                        fv = parse_ru_number(row.iloc[idx], 0.0)  # #12: запятая/неразрывный пробел/пусто→0
                        if q == 'Y':
                            annual_map.setdefault(yr, {'plan': 0.0, 'forecast': 0.0, 'fact': 0.0})[kind] = fv
                        else:
                            periods.setdefault((yr, q), {'plan': 0.0, 'forecast': 0.0, 'fact': 0.0})[kind] = fv
                    item['periods'] = [
                        {'year': yr, 'q': q,
                         'plan': v['plan'],
                         'forecast': v['forecast'] if v['forecast'] else v['plan'],
                         'fact': v['fact']}
                        for (yr, q), v in sorted(periods.items())
                    ]
                    # годовой план по каждому году (если в таблице есть годовой блок)
                    item['annual'] = {int(yr): {'plan': v['plan'],
                                                'forecast': v['forecast'] if v['forecast'] else v['plan'],
                                                'fact': v['fact']}
                                      for yr, v in sorted(annual_map.items())}
                    data.append(item)
                return pd.DataFrame(data)
            else:
                # ID-столбец читаем КАК СТРОКУ: иначе pandas приводит «№ п/п» к float и
                # «1.10» становится «1.1» (теряется значащий ноль), ломая иерархию «1.10.1».
                try:
                    df = pd.read_excel(filepath, dtype={ScheduleColumnMap.ID: str})
                except Exception:
                    df = pd.read_excel(filepath)
                df.columns = df.columns.str.strip()
                return df
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel: {e}")
            return pd.DataFrame()

    @classmethod
    def build_system_context(cls, schedule_path: str, indicators_path: str, methodologies_dir: str, finances_path: str = None):
        df_schedule = cls.safe_load_excel(schedule_path, is_indicator=False)
        df_indicators = cls.safe_load_excel(indicators_path, is_indicator=True)
        
        # Парсим финансы
        finances_data = cls.safe_load_finances(finances_path) if finances_path else {'mapped': {}}
        fin_mapped = finances_data['mapped']
        unmatched_finances = []
        
        if df_schedule.empty: raise ValueError("План-график пуст.")

        if not df_indicators.empty and IndicatorColumnMap.METHODOLOGY in df_indicators.columns:
            MethodologyParser.preload_methodologies(df_indicators[IndicatorColumnMap.METHODOLOGY].unique().tolist(), methodologies_dir)

        raw_nodes, edges, temp_dag = {}, [], nx.DiGraph()
        for _, row in df_schedule.iterrows():
            if pd.isna(row.get(ScheduleColumnMap.ID)): continue
            node_id = str(row[ScheduleColumnMap.ID]).strip()
            if node_id.endswith('.0'):
                node_id = node_id[:-2]
            raw_nodes[node_id] = {
                'type': str(row.get(ScheduleColumnMap.TYPE, 'Task')).strip(),
                'name': str(row.get(ScheduleColumnMap.NAME, '')).strip(),
                'F': clean_float(row.get(ScheduleColumnMap.BUDGET, 0.0)),
                'start': pd.to_datetime(row.get(ScheduleColumnMap.START_DATE), errors='coerce'),
                'end': pd.to_datetime(row.get(ScheduleColumnMap.END_DATE), errors='coerce')
            }
            temp_dag.add_node(node_id)
            parts = [p for p in node_id.split('.') if p]
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                edges.append({'source': node_id, 'target': parent_id})
                temp_dag.add_edge(node_id, parent_id)
                
        # --- ШАГ 1: Маппинг финансов к raw_nodes ДО расчета дат ---
        schedule_ids = set(raw_nodes.keys())
        for f_id, f_data in fin_mapped.items():
            if f_id not in schedule_ids:
                unmatched_finances.append({'id': f_id, 'name': f_data['name'], 'profile': f_data['profile']})
                
        for n_id in raw_nodes.keys():
            raw_nodes[n_id]['finances'] = fin_mapped.get(n_id, {}).get('profile', {})

        # --- Расчет дат (унаследовано из старого кода) ---
        nodes = []
        today_dt = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))

        def inherited_start(nid):
            cur = nid
            while True:
                s = raw_nodes.get(cur, {}).get('start')
                if pd.notna(s):
                    return s
                parts = [p for p in cur.split('.') if p]
                if len(parts) <= 1:
                    return today_dt
                cur = ".".join(parts[:-1])

        def direct_children(pid):
            try: return list(temp_dag.predecessors(pid))
            except Exception: return []

        resolved = {}
        order = sorted(raw_nodes.keys(), key=lambda x: -len([p for p in str(x).split('.') if p]))
        for n_id in order:
            is_milestone = str(raw_nodes[n_id]['type']).strip().lower() in ('веха', 'milestone')
            st = raw_nodes[n_id]['start']
            en = raw_nodes[n_id]['end']
            if is_milestone:
                if pd.isna(en):
                    en = st if pd.notna(st) else inherited_start(n_id)
                st = en
            else:
                kids = direct_children(n_id)
                if kids:
                    ks = [resolved[k][0] for k in kids if k in resolved and pd.notna(resolved[k][0])]
                    ke = [resolved[k][1] for k in kids if k in resolved and pd.notna(resolved[k][1])]
                    if pd.isna(st) and ks: st = min(ks)
                    if pd.isna(en) and ke: en = max(ke)
                if pd.isna(st): st = inherited_start(n_id)
                if pd.isna(en): en = st
            resolved[n_id] = (st, en)

        # --- ШАГ 1: Формирование финального списка узлов (nodes) ---
        for n_id in raw_nodes.keys():
            st, en = resolved[n_id]
            
            # Начальное значение F: ТОЛЬКО из финансового профиля (отдельная таблица финансов).
            # Бюджет из колонки план-графика («Бюджет (млн. руб.)») НИГДЕ не используется как деньги —
            # план-график задаёт только сроки/структуру. Нет профиля → F=0 (при пересчёте
            # _recompute_leaf_values это же значение подтвердится, риск-взвешенный f_eff заменит его,
            # если профиль появится позже).
            f_total = 0.0
            fin_prof = raw_nodes[n_id].get('finances', {})
            if fin_prof:
                for y, vals in fin_prof.items():
                    f_total += vals.get('base', 0.0)

            nodes.append({
                'id': n_id, 'type': raw_nodes[n_id]['type'], 'name': raw_nodes[n_id]['name'],
                'F': f_total, 
                'finances': json.dumps(fin_prof, ensure_ascii=False),
                # ФИНАНСОВАЯ СУЩНОСТЬ: узел присутствует в таблице финансов → на него (и только
                # на таких) распределяются деньги родителей, пропорционально весу. Нефинансовые
                # вехи денег при распределении не получают. Флаг можно включить вручную (чек-бокс
                # «Финансовая веха» в план-графике) или он включится сам при вводе денег.
                'is_financial': bool(fin_prof),
                'T_start': st.strftime('%Y-%m-%d'), 'T_end': en.strftime('%Y-%m-%d'),
                'T_opt': max(1, (en - st).days)
            })

        methodologies_dict = {}
        # ... (дальше идет код парсинга показателей)
        if not df_indicators.empty:
            # Группируем показатели по уникальной паре (название, методика)
            df_indicators['group_key'] = df_indicators[IndicatorColumnMap.NAME].astype(str) + '|||' + df_indicators[IndicatorColumnMap.METHODOLOGY].astype(str)
            for gkey, group in df_indicators.groupby('group_key'):
                kpi_name = group.iloc[0][IndicatorColumnMap.NAME]
                meth_file = group.iloc[0][IndicatorColumnMap.METHODOLOGY]
                task_ids = group[IndicatorColumnMap.TASK_ID].unique().tolist()
                valid_tasks = [t for t in task_ids if t in temp_dag]  # только существующие в графе
                if not valid_tasks:
                    continue

                row = group.iloc[0]
                q1, q2, q3, q4, y = row.get('Q1_plan', 0), row.get('Q2_plan', 0), row.get('Q3_plan', 0), row.get('Q4_plan', 0), row.get('Year_plan', 0)
                sum_q = q1 + q2 + q3 + q4
                weights = [q1/sum_q, q2/sum_q, q3/sum_q, q4/sum_q] if sum_q > 0 else [0.25, 0.25, 0.25, 0.25]

                # ДОБАВЛЕНО: мультигодовые квартальные периоды показателя.
                periods = row.get('periods', None)
                if not isinstance(periods, list) or not periods:
                    # Фолбэк: одна годовая четвёрка кварталов. Год берём из дат план-графика,
                    # иначе — текущий год.
                    base_year = None
                    valid_starts = [raw_nodes[t]['start'] for t in valid_tasks if t in raw_nodes and pd.notna(raw_nodes[t]['start'])]
                    if valid_starts:
                        base_year = int(min(valid_starts).year)
                    base_year = base_year or datetime.now().year
                    periods = [
                        {'year': base_year, 'q': 1, 'plan': float(q1), 'forecast': float(q1), 'fact': 0.0},
                        {'year': base_year, 'q': 2, 'plan': float(q2), 'forecast': float(q2), 'fact': 0.0},
                        {'year': base_year, 'q': 3, 'plan': float(q3), 'forecast': float(q3), 'fact': 0.0},
                        {'year': base_year, 'q': 4, 'plan': float(q4), 'forecast': float(q4), 'fact': 0.0},
                    ]
                base_year = min(int(p['year']) for p in periods)

                # ДОБАВЛЕНО (#1): годовой план по каждому году — отдельно, НЕ как сумма кварталов.
                annual = row.get('annual', None)
                if not isinstance(annual, dict) or not annual:
                    annual = {base_year: {'plan': parse_ru_number(y), 'forecast': parse_ru_number(y), 'fact': 0.0}}
                annual = {int(yr): v for yr, v in annual.items()}
                # на всякий случай гарантируем годовой план для каждого года из периодов
                for p in periods:
                    annual.setdefault(int(p['year']), {'plan': 0.0, 'forecast': 0.0, 'fact': 0.0})

                kpi_id = f"KPI_{hashlib.md5(gkey.encode('utf-8')).hexdigest()[:8]}"
                nodes.append({
                    'id': kpi_id, 'type': 'KPI', 'name': kpi_name, 'F': 0.0,
                    'T_start': f'{base_year}-01-01', 'T_end': f'{max(int(p["year"]) for p in periods)}-12-31', 'T_opt': 0,
                    'Q1': q1, 'Q2': q2, 'Q3': q3, 'Q4': q4, 'Year': annual.get(base_year, {}).get('plan', parse_ru_number(y)),
                    'base_year': base_year,
                    'periods': json.dumps(periods, ensure_ascii=False),
                    'annual': json.dumps({str(yr): v for yr, v in annual.items()}, ensure_ascii=False),
                    'quarter_weights': json.dumps(weights)
                })
                for t in valid_tasks:
                    edges.append({'source': t, 'target': kpi_id})
                methodologies_dict[kpi_id] = MethodologyParser.get_text(meth_file, methodologies_dir)

        return pd.DataFrame(nodes), pd.DataFrame(edges), methodologies_dict, unmatched_finances

# ==========================================
# 4. ДВИЖОК ИИ С ЗАЩИТОЙ ОТ ПАРСИНГА
# ==========================================
class LocalLLMEngine:
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "local", model_name: str = "gpt-oss:20b",
                 use_meth_cache: bool = True, meth_cache_dir: str = "data/meth_cache", force_recompress: bool = False,
                 timeout: float = 120.0, enabled: bool = True):
        self.enabled = bool(enabled)
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = float(timeout) if timeout else 120.0
        # api_key обязателен для OpenAI-клиента; для локальной Ollama подойдёт любая строка.
        self.client = OpenAI(base_url=base_url, api_key=(api_key or "local"), timeout=self.timeout)
        self.use_meth_cache = use_meth_cache
        self.meth_cache_dir = meth_cache_dir
        self.force_recompress = force_recompress
        os.makedirs(self.meth_cache_dir, exist_ok=True)

    def test_connection(self) -> Tuple[bool, str]:
        """Быстрая проверка доступности модели. Возвращает (успех, сообщение)."""
        if not self.enabled:
            return (False, "ИИ отключён в настройках.")
        try:
            with _LLM_GATE:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    temperature=0.0, max_tokens=1,
                )
            _ = resp.choices[0].message.content
            return (True, f"Подключение успешно · модель «{self.model_name}»")
        except Exception as e:
            msg = str(e)
            return (False, f"Не удалось подключиться: {msg[:200]}")

    def _get_cache_filepath(self, meth_hash: str) -> str:
        return os.path.join(self.meth_cache_dir, f"{meth_hash}.json")

    def _load_from_cache(self, meth_hash: str) -> Optional[str]:
        if not self.use_meth_cache:
            return None
        filepath = self._get_cache_filepath(meth_hash)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Проверяем: не устарела ли запись (7 дней), не требуется ли пересжатие, и СОВПАДАЕТ ЛИ
            # МОДЕЛЬ — иначе при смене ИИ-модели отдавался бы текст, сжатый прежней моделью.
            fresh = (time.time() - data.get("timestamp", 0) < 604800) and not self.force_recompress
            same_model = (data.get("model") == self.model_name)
            if fresh and same_model:
                return data.get("text")
            return None
        except Exception:
            return None

    def _save_to_cache(self, meth_hash: str, text: str):
        if not self.use_meth_cache:
            return
        filepath = self._get_cache_filepath(meth_hash)
        data = {
            "text": text,
            "timestamp": time.time(),
            "model": self.model_name
        }
        try:
            _atomic_write_text(filepath, json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Не удалось сохранить кэш методики {meth_hash}: {e}")

    # ---- Поэтапное сохранение выжимок чанков (возобновление после прерывания, п.3) ----
    def _chunk_cache_path(self, meth_hash: str) -> str:
        return os.path.join(self.meth_cache_dir, f"{meth_hash}.chunks.json")

    def _load_chunk_progress(self, meth_hash: str, n_chunks: int) -> Dict[int, str]:
        if not self.use_meth_cache:
            return {}
        path = self._chunk_cache_path(meth_hash)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('n_chunks') != n_chunks:  # нарезка изменилась — прогресс не валиден
                return {}
            return {int(k): v for k, v in data.get('summaries', {}).items()}
        except Exception:
            return {}

    def _save_chunk_progress(self, meth_hash: str, n_chunks: int, summaries: Dict[int, str]):
        if not self.use_meth_cache:
            return
        try:
            payload = {'n_chunks': n_chunks, 'summaries': {str(k): v for k, v in summaries.items()}}
            _atomic_write_text(self._chunk_cache_path(meth_hash), json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Не удалось сохранить прогресс сжатия методики: {e}")

    @staticmethod
    def _split_methodology(text: str, max_len: int = 2500) -> List[str]:
        """РЕАЛИЗОВАНО (п.1): нарезка методики НЕ вслепую по длине, а по границам абзацев и
        предложений — слово не рвётся, смысл фрагмента сохраняется. Алгоритм: режем на абзацы;
        слишком длинный абзац — на предложения; затем «упаковываем» части в чанки до max_len,
        не разрывая предложения (и только в крайнем случае — очень длинное предложение по словам)."""
        paras = [p.strip() for p in re.split(r'\n\s*\n', text or "") if p.strip()]
        units: List[str] = []
        for p in paras:
            if len(p) <= max_len:
                units.append(p)
                continue
            sents = re.split(r'(?<=[.!?;])\s+', p)
            buf = ""
            for s in sents:
                if len(s) > max_len:  # аномально длинное предложение — режем по словам
                    if buf:
                        units.append(buf); buf = ""
                    wbuf = ""
                    for w in s.split(" "):
                        if wbuf and len(wbuf) + len(w) + 1 > max_len:
                            units.append(wbuf); wbuf = w
                        else:
                            wbuf = (wbuf + " " + w).strip()
                    if wbuf:
                        buf = wbuf
                elif buf and len(buf) + len(s) + 1 > max_len:
                    units.append(buf); buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                units.append(buf)
        # упаковываем части в чанки по границам, не разрывая предложения/абзацы
        chunks: List[str] = []
        cur = ""
        for u in units:
            if cur and len(cur) + len(u) + 2 > max_len:
                chunks.append(cur); cur = u
            else:
                cur = (cur + "\n\n" + u).strip()
        if cur:
            chunks.append(cur)
        return chunks or [(text or "")[:max_len]]

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, tag: str = "LLM") -> str:
        """Единая точка вызова LLM (system + user) с логированием запроса и ответа (п.11 методики)."""
        logger.info(f"\n[LLM {tag} PROMPT]\n--- SYSTEM ---\n{system_prompt}\n--- USER ---\n{user_prompt}")
        if not getattr(self, 'enabled', True):
            # ИИ выключен в настройках — вызывающий код переходит на математические дефолты/кеш.
            raise RuntimeError("LLM disabled by settings")
        # Глобальная очередь: к Ollama идём по одному (исключает гонки нагрузки и таймауты).
        with _LLM_GATE:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        out = (resp.choices[0].message.content or "").strip()
        logger.info(f"[LLM {tag} RESPONSE]\n{out}")
        return out

    @staticmethod
    def _extract_json(text: str):
        """Извлекает JSON-объект/массив из ответа модели и парсит его (#2).

        Снимает markdown ```...```, вырезает блоки рассуждений <think>…</think> (частые у
        reasoning-моделей вроде gpt-oss), выбирает последний сбалансированный {…}/[…],
        экранирует одиночные обратные слеши. Пустой ответ → понятная ошибка для фолбэка."""
        if text is None:
            raise ValueError("пустой ответ модели")
        t = str(text).strip()
        if not t:
            raise ValueError("пустой ответ модели")
        # вырезаем reasoning-блоки
        t = re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r'<\|.*?\|>', '', t, flags=re.DOTALL)  # спец-токены
        t = re.sub(r'```[a-zA-Z]*\s*', '', t)
        t = t.replace('```', '').strip()
        if not t:
            raise ValueError("после очистки ответ пуст")
        # последний сбалансированный объект — обычно это финальный ответ
        obj = list(re.finditer(r'\{.*?\}', t, re.DOTALL))
        candidate = None
        if obj:
            # берём самый «жадный» вариант: от первой { до последней }
            first = t.find('{'); last = t.rfind('}')
            candidate = t[first:last + 1] if (first != -1 and last > first) else obj[-1].group(0)
        else:
            arr_first, arr_last = t.find('['), t.rfind(']')
            candidate = t[arr_first:arr_last + 1] if (arr_first != -1 and arr_last > arr_first) else t
        candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
        return json.loads(candidate)

    def compress_methodology(self, meth_text: str) -> str:
        """Сжимает объёмную методику KPI до расчётного конспекта (map-reduce через LLM).

        Улучшения: (п.1) чанки режутся по абзацам/предложениям; (п.3) выжимка каждого чанка
        сохраняется СРАЗУ — при прерывании расчёт продолжится с того же места, а не сначала;
        объединение (reduce) выполняется всегда, когда чанков больше одного."""
        if not meth_text or len(meth_text.strip()) < 1500:
            return meth_text
        meth_hash = hashlib.md5(meth_text.encode('utf-8')).hexdigest()

        cached = self._load_from_cache(meth_hash)
        if cached is not None:
            return cached

        chunk_sys = (
            "Ты — методолог-аналитик. Из фрагмента официальной методики расчёта KPI извлекай "
            "ТОЛЬКО то, что нужно для воспроизведения расчёта: точные формулы; обозначения и "
            "единицы переменных; веса и коэффициенты; пороги и условия; периодичность "
            "(кварталы/год); источники данных. Не пересказывай вводную часть, ничего не "
            "придумывай и не добавляй от себя. Если расчётной информации во фрагменте нет — "
            "верни пустую строку."
        )
        chunks = self._split_methodology(meth_text, max_len=2500)  # п.1: по абзацам/предложениям
        progress = self._load_chunk_progress(meth_hash, len(chunks))  # п.3: что уже посчитано
        for i, chunk in enumerate(chunks):
            if i in progress:
                continue  # этот чанк уже сжат в прошлый раз — пропускаем
            user = (f"Фрагмент {i + 1} из {len(chunks)} методики:\n<<<\n{chunk}\n>>>\n"
                    f"Верни компактную выжимку маркированным списком; формулы приводи как есть.")
            try:
                summary = self._chat(chunk_sys, user, temperature=0.1, tag=f"METH COMPRESS {i + 1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Ошибка при сжатии чанка {i + 1}: {e}")
                summary = ""
            progress[i] = summary or ""
            self._save_chunk_progress(meth_hash, len(chunks), progress)  # п.3: пишем сразу

        chunk_summaries = [progress[i] for i in sorted(progress) if progress[i].strip()]

        if not chunk_summaries:
            result = meth_text[:2000]
        elif len(chunk_summaries) == 1:
            result = chunk_summaries[0]  # один содержательный фрагмент — объединять нечего
        else:
            # ОБЪЕДИНЕНИЕ (reduce) — выполняется всегда при нескольких выжимках.
            reduce_sys = (
                "Ты — методолог-аналитик. Объедини разрозненные выжимки в единый расчётный конспект "
                "методики KPI без потери формул, переменных, весов, порогов и условий. Удали повторы, "
                "сохрани все числовые параметры. Структурируй ответ по разделам: "
                "1) Итоговая формула; 2) Переменные и единицы; 3) Веса и коэффициенты; "
                "4) Условия и пороги; 5) Периодизация."
            )
            reduce_user = "Выжимки фрагментов методики:\n" + "\n---\n".join(chunk_summaries)
            try:
                result = self._chat(reduce_sys, reduce_user, temperature=0.1, tag="METH REDUCE")
            except Exception as e:
                logger.error(f"Ошибка reduce методики: {e}")
                result = "\n".join(chunk_summaries)[:2000]

        self._save_to_cache(meth_hash, result)
        # прогресс по чанкам больше не нужен — финальный конспект закеширован
        try:
            cp = self._chunk_cache_path(meth_hash)
            if os.path.exists(cp):
                os.remove(cp)
        except Exception:
            pass
        return result

    def propose_kpi_calibration(self, kpi_name: str, methodology: str) -> Optional[Dict[str, Any]]:
        """По методике предлагает калибровку точности KPI И классифицирует связь с планом.

        Ключевая проблема: формула KPI часто опирается на ВНЕШНИЕ ИСХОДЫ (средний балл
        поступающих), которые план меняет лишь косвенно (ремонт корпуса → привлекательность →
        балл). Поэтому модель различает:
          • link_type='прямой'  — хотя бы одна переменная формулы производится планом (формула применима);
          • link_type='косвенный' — переменные формулы внешние, связь с планом — допущение (низкая уверенность).
        Возвращает driver/sensitivity/confidence/link_type/formula/rationale; None при сбое/офлайн."""
        meth = self.compress_methodology(methodology or "")
        if not meth or len(meth.strip()) < 30:
            return None
        system_prompt = (
            "Ты — методолог. По методике расчёта KPI сделай две вещи. "
            "(1) Извлеки итоговую ФОРМУЛУ KPI и её переменные. "
            "(2) Оцени, насколько переменные формулы УПРАВЛЯЮТСЯ мероприятиями проекта (бюджет/сроки), "
            "а не внешними факторами. Если хотя бы одна переменная формулы напрямую производится "
            "мероприятиями — связь 'прямой'; если переменные — внешние исходы (баллы, удовлетворённость, "
            "доля рынка), на которые план влияет лишь опосредованно — связь 'косвенный'. "
            "Верни СТРОГО JSON без пояснений: "
            '{"formula":"...","driver":"бюджет|срок|оба","link_type":"прямой|косвенный",'
            '"sensitivity":<0.5..1.5>,"confidence":<0.3..1.0>,"rationale":"как план связан с KPI, кратко по-русски"}. '
            "Для 'косвенный' ставь confidence ниже (0.3–0.6) и обычно sensitivity ≤1: связь слабее и неопределённее."
        )
        user_prompt = f"KPI: {kpi_name}\nМетодика (конспект):\n<<<\n{meth[:2500]}\n>>>"
        try:
            raw = self._chat(system_prompt, user_prompt, temperature=0.1, tag=f"KPI CALIB {kpi_name[:30]}")
            data = self._extract_json(raw)
            if not isinstance(data, dict):
                return None
            drv = str(data.get('driver', 'оба')).strip().lower()
            if drv not in ('бюджет', 'срок', 'оба'):
                drv = 'оба'
            link = str(data.get('link_type', '')).strip().lower()
            if link not in ('прямой', 'косвенный'):
                link = 'косвенный'  # по умолчанию консервативно — связь считаем косвенной
            sens = float(data.get('sensitivity', 1.0))
            conf = float(data.get('confidence', 1.0))
            # Косвенная связь не должна выдавать высокую уверенность даже при оптимизме модели.
            if link == 'косвенный':
                conf = min(conf, 0.6)
            return {
                'driver': drv,
                'link_type': link,
                'formula': str(data.get('formula', ''))[:400],
                'sensitivity': float(min(1.5, max(0.5, sens))),
                'confidence': float(min(1.0, max(0.3, conf))),
                'rationale': str(data.get('rationale', ''))[:400],
            }
        except Exception as e:
            logger.warning(f"propose_kpi_calibration({kpi_name}): {e}")
            return None

    def extract_semantic_weights_for_kpi(self, edges_df: pd.DataFrame, methodologies: Dict, nodes_df: pd.DataFrame, kpi_id: str, subgraph_nodes: set) -> Dict:
        node_info = nodes_df.set_index('id')[['name', 'type']].to_dict('index')
        meth_text = methodologies.get(kpi_id, "")
        meth_compressed = self.compress_methodology(meth_text)
        kpi_name = node_info.get(kpi_id, {}).get('name', kpi_id)

        system_prompt = (
            "Ты — эксперт по проектному управлению и декомпозиции KPI. Оцени, какой вклад каждая "
            "дочерняя работа вносит в выполнение родительской задачи С ТОЧКИ ЗРЕНИЯ влияния на "
            "заданный KPI и его методику расчёта.\n"
            "Правила:\n"
            "- weight — доля влияния в диапазоне [0, 1]; сумма всех weight строго равна 1.0.\n"
            "- Опирайся на методику KPI и смысл работ, а не на их количество: работы, прямо "
            "входящие в формулу KPI, весят больше вспомогательных.\n"
            "- Для каждой работы укажи ТИП зависимости (type) — форму её влияния на родителя:\n"
            "    • linear — пропорциональный вклад (по умолчанию, если нет явных причин для иного);\n"
            "    • saturating — убывающая отдача: сверх достаточного объёма вклад почти не растёт;\n"
            "    • threshold — эффект появляется только после достижения критической массы;\n"
            "    • amplifying — синергия/мультипликатор: сильный вклад усиливается;\n"
            "    • inhibitory — работа СНИЖАЕТ показатель (риск, ограничение, конкуренция за ресурс).\n"
            "  Если уверенности нет — ставь linear.\n"
            "- rationale пиши КРАТКО и ТОЛЬКО НА РУССКОМ ЯЗЫКЕ (без других языков, без транслита).\n"
            "- Используй ТОЛЬКО перечисленные работы — ничего не добавляй и не пропускай.\n"
            "- Ответ — строго ОДИН JSON-объект, без markdown и без текста вне JSON."
        )

        edge_attrs = {}
        for target, group in edges_df.groupby('target'):
            children_ids = [str(c) for c in group['source'].tolist()]
            if not children_ids:
                continue

            # Методика, п.6: единственный потомок или связь к KPI → вес 1.0 без обращения к LLM.
            target_is_kpi = str(node_info.get(target, {}).get('type', '')).upper() == 'KPI'
            if target_is_kpi or len(children_ids) == 1:
                for cid in children_ids:
                    edge_attrs[(cid, str(target))] = {'weight': 1.0, 'relation_type': 'direct', 'rationale': 'единственная/прямая связь'}
                continue

            target_name = node_info.get(target, {}).get('name', target)
            children_lines = "\n".join(
                f"- {cid} — {node_info.get(cid, {}).get('name', cid)}" for cid in children_ids
            )
            user_prompt = (
                f"KPI: {kpi_name}\n"
                f"Методика расчёта (конспект):\n"
                f"{meth_compressed or '(методика не предоставлена — оцени по смыслу названий работ)'}\n\n"
                f"Родительская задача: {target} «{target_name}»\n"
                f"Дочерние работы:\n{children_lines}\n\n"
                f"Верни JSON строго по схеме (включи ВСЕ работы; сумма weight = 1.0; "
                f"type ∈ [linear, saturating, threshold, amplifying, inhibitory]; rationale только по-русски):\n"
                f'{{"weights": [{{"source": "<id>", "weight": <число>, '
                f'"type": "<linear|saturating|threshold|amplifying|inhibitory>", '
                f'"rationale": "<кратко по-русски, до 10 слов>"}}]}}'
            )
            try:
                text = self._chat(system_prompt, user_prompt, temperature=0.1, tag=f"WEIGHT KPI {kpi_id} -> {target}")
                parsed = self._extract_json(text)
                items = parsed.get('weights', []) if isinstance(parsed, dict) else parsed
                total_w = sum(float(it.get('weight', 0.0)) for it in items)
                seen = set()
                for it in items:
                    src = str(it.get('source'))
                    if src not in children_ids:
                        continue  # игнорируем «придуманные» моделью узлы
                    rt = canonical_relation(it.get('type'))  # тип зависимости от LLM
                    edge_attrs[(src, str(target))] = {
                        'weight': round(float(it.get('weight', 0.0)) / total_w, 4) if total_w > 0 else round(1.0 / len(children_ids), 4),
                        'relation_type': rt,
                        'rationale': str(it.get('rationale', ''))[:120],
                    }
                    seen.add(src)
                # Защита от пропусков: не вернувшиеся работы делят оставшийся вес поровну.
                missing = [c for c in children_ids if c not in seen]
                if missing:
                    assigned = sum(edge_attrs[(s, str(target))]['weight'] for s in seen)
                    share = round(max(0.0, 1.0 - assigned) / len(missing), 4)
                    for s in missing:
                        edge_attrs[(s, str(target))] = {'weight': share, 'relation_type': 'semantic', 'rationale': 'дополнено равным остатком'}
            except Exception as e:
                logger.error(f"Ошибка извлечения весов для KPI {kpi_id}, target {target}: {e}. Использую равные веса.")
                for src in children_ids:
                    edge_attrs[(src, str(target))] = {'weight': round(1.0 / len(children_ids), 4), 'relation_type': 'equal', 'rationale': 'fallback: равные веса'}
        return edge_attrs

    def generate_impact_report(self, target_context: str, kpi_name: str, pct_change: float,
                               weight=None, influence=None, periods=None, annual=None,
                               meth_text: str = "") -> Dict[str, Any]:
        """ИИ ОПРЕДЕЛЯЕТ новые значения показателя по кварталам.

        Ему передаются: ВЕС сущности в графе, её ВЛИЯНИЕ на показатель, КАК изменился
        показатель (итоговый %), и план по кварталам (с пометкой закрытых). Он возвращает
        новые значения по ОТКРЫТЫМ кварталам ("values") и текстовое пояснение.
        Жёсткие ограждения (закрытые кварталы, «не выше плана при снижении») накладываются
        в приложении поверх ответа — здесь только просим их соблюдать."""
        compressed_meth = self.compress_methodology(meth_text)
        periods = periods or []
        annual = annual or {}

        if pct_change >= 0:
            event_text = f"изменение ПОВЫСИЛО показатель на {pct_change * 100:.1f}%"
            rule_dir = "Поскольку показатель ВЫРОС, ни один открытый квартал НЕ может быть НИЖЕ своего плана."
        else:
            event_text = f"изменение СНИЗИЛО показатель на {abs(pct_change) * 100:.1f}%"
            rule_dir = "Поскольку показатель СНИЗИЛСЯ, ни один открытый квартал НЕ может быть ВЫШЕ своего плана (даже на чуть-чуть)."

        w_txt = f"{float(weight):.3f}" if weight is not None else "(не задан)"
        i_txt = f"{float(influence) * 100:.0f}%" if influence is not None else "(не задано)"

        q_lines, open_labels = [], []
        for p in periods:
            q_lines.append(f"{p.get('label','')}: план {p.get('plan',0):.2f} "
                           f"(ориентир модели: {p.get('forecast',0):.2f})")
            open_labels.append(p.get('label', ''))
        y_lines = [f"Год {y}: план {a.get('plan',0):.2f} → модель даёт {a.get('forecast',0):.2f}"
                   for y, a in sorted(annual.items())]
        facts = "\n".join(q_lines + y_lines) or "(нет числовых данных)"

        system_prompt = (
            "Ты — риск-аналитик портфеля. По данным о сущности (её вес и влияние на показатель) "
            "и о том, как изменился показатель в целом, ты ОПРЕДЕЛЯЕШЬ новые значения показателя "
            "по кварталам и коротко их поясняешь.\n"
            "Строгие правила:\n"
            "- Верни новое значение для КАЖДОГО квартала (число).\n"
            f"- {rule_dir}\n"
            "- Значения по кварталам в среднем должны отражать итоговое изменение показателя "
            "(масштаб задаёт «ориентир модели»); можешь перераспределять эффект между кварталами по "
            "логике методики (нарастание/убывание/сезонность), но НЕ разворачивать общий знак.\n"
            "- Числа без единиц и разделителей тысяч, десятичный разделитель — точка.\n"
            "- text_report: 2–4 предложения по-русски, простыми словами, без формул и LaTeX.\n"
            "- Ответ — строго ОДИН JSON-объект без markdown."
        )
        user_prompt = (
            f"Источник изменения: {target_context}\n"
            f"Вес сущности в графе: {w_txt}\n"
            f"Влияние сущности на показатель: {i_txt}\n"
            f"Показатель: {kpi_name}\n"
            f"Итоговое изменение показателя: {pct_change * 100:+.1f}% ({event_text}).\n"
            f"Методика (конспект): {compressed_meth or '(не предоставлена)'}\n\n"
            f"Значения по периодам:\n{facts}\n\n"
            f'Схема ответа: {{"values": {{"<кв.>": <число>, ...}}, "text_report": "<текст>"}}'
        )
        try:
            raw_text = self._chat(system_prompt, user_prompt, temperature=0.3, tag="IMPACT")
            data = self._extract_json(raw_text)
            raw_vals = data.get('values', {}) or {}
            values = {}
            for k, v in raw_vals.items():
                fv = parse_ru_number(v, None)
                if fv is not None:
                    values[str(k)] = float(fv)
            return {"values": values, "text_report": data.get('text_report', '') or ""}
        except Exception as e:
            logger.warning(f"ИИ-отчёт: не удалось разобрать JSON ({e}); значения по модели.")
            direction = "вырос" if pct_change >= 0 else "снизился"
            return {"values": {},  # пусто → приложение оставит модельные значения
                    "text_report": f"Показатель {direction} на {abs(pct_change)*100:.1f}% относительно плана. "
                                   f"Закрытые кварталы остаются на плане, эффект — в открытых периодах."}

# ==========================================
# 5. ЯДРО СИМУЛЯТОРА «МИКШЕР»
# ==========================================
def portfolio_greedy_allocation(engines: Dict[str, "ProjectMixer"], extra_budget: float,
                                step: float = 5.0, max_steps: int = 60) -> Dict[str, Any]:
    """Межпроектное распределение ДОПОЛНИТЕЛЬНОГО бюджета по портфелю проектов.

    Жадно: на каждом шаге `step` млн идут в ту (проект, работу), где **относительный** прирост
    корзины KPI проекта наибольший (относительность делает проекты сопоставимыми). Учитывается
    убывающая отдача (шаг реально применяется, следующий ищется заново). Неразрушающе —
    бюджеты восстанавливаются в конце; возвращается план распределения и итоговые приросты."""
    if not engines or extra_budget <= 0 or step <= 0:
        return {'allocations': [], 'project_gain': {}, 'spent': 0.0}

    def kpi_sum(eng):
        return sum(eng._kpi_value(k) for k in eng.kpi_ids) or 0.0

    # extra_budget и step заданы в ОБЩЕЙ единице (млн). Для каждого проекта переводим шаг
    # в его собственную единицу: млн / factor (тыс → ×1000, руб → ×1e6), чтобы добавлять
    # к F, который хранится в единице проекта. Так разные единицы становятся сопоставимы.
    factor = {slug: BUDGET_UNITS.get(getattr(eng.config, 'budget_scale', 'millions'),
                                     BUDGET_UNITS['millions'])['factor']
              for slug, eng in engines.items()}
    step_proj = {slug: (step / factor[slug] if factor[slug] else step) for slug in engines}

    # Снимок ПОЛНОГО финансового состояния: деньги живут в finances (F — производная величина),
    # поэтому снапшота одного F недостаточно для честного отката.
    def _snap(eng):
        return {n: {'F': eng.G.nodes[n].get('F', 0.0),
                    'T_end': eng.G.nodes[n].get('T_end', ''),
                    'local_value': eng.G.nodes[n].get('local_value', 0.0),
                    'finances': copy.deepcopy(eng.G.nodes[n].get('finances', {})),
                    'finances_eff': copy.deepcopy(eng.G.nodes[n].get('finances_eff', None)),
                    'is_financial': eng.G.nodes[n].get('is_financial', False)}
                for n in eng.G.nodes()
                if str(eng.G.nodes[n].get('type', '')).upper() != 'KPI'}

    def _restore(eng, snap):
        for n, s in snap.items():
            eng.G.nodes[n].update(s)
        eng._compute_effective_finances(); eng._recompute_leaf_values()
        eng._recompute_parent_budgets(); eng._propagate_all_kpis()

    def _recalc(eng):
        eng._compute_effective_finances(); eng._recompute_leaf_values()
        eng._recompute_parent_budgets(); eng._propagate_all_kpis()

    snaps = {slug: _snap(eng) for slug, eng in engines.items()}
    # деньги можно давать только ФИНАНСОВЫМ работам (организационные вехи бюджет не осваивают)
    cand_leaves = {slug: [L for L in eng.get_leaves() if eng._is_financial_leaf(L)]
                   for slug, eng in engines.items()}
    # фиксируем текущее распределение, чтобы правка одной работы не перетягивалась целью родителя
    for slug, eng in engines.items():
        if cand_leaves[slug]:
            eng._clear_rollup_sources(cand_leaves[slug][0])
            _recalc(eng)
    base_sum = {slug: kpi_sum(eng) for slug, eng in engines.items()}
    alloc: Dict[Tuple[str, str], float] = {}
    spent = 0.0
    result, project_gain = [], {}
    try:
        n_steps = min(int(extra_budget // step), max_steps)
        for _ in range(n_steps):
            best, best_gain = None, 1e-9
            for slug, eng in engines.items():
                b = kpi_sum(eng) or 1e-9
                sp = step_proj[slug]
                for L in cand_leaves[slug]:
                    a = eng.G.nodes[L]
                    try:
                        r = eng.mix(L, float(a.get('F', 0.0)) + sp, a['T_start'], a['T_end'], project=False)
                    except Exception:
                        continue
                    new_sum = sum(r[k]['new'] for k in eng.kpi_ids)
                    gain = (new_sum - b) / b  # относительный прирост корзины KPI проекта
                    if gain > best_gain:
                        best_gain, best = gain, (slug, L)
            if not best:
                break
            slug, L = best
            eng = engines[slug]
            # ДЕНЬГИ ПИШЕМ В ФИНАНСЫ (а не в F: F пересчитывается из финансов и присвоение стёрлось бы)
            new_F = float(eng.G.nodes[L].get('F', 0.0)) + step_proj[slug]
            eng.G.nodes[L]['finances'] = json.dumps(eng._finances_for_nominal(L, new_F), ensure_ascii=False)
            eng.G.nodes[L]['is_financial'] = True
            _recalc(eng)
            alloc[(slug, L)] = alloc.get((slug, L), 0.0) + step   # учёт в ОБЩЕЙ единице (млн)
            spent += step
        # итог: приросты по проектам при выбранном распределении
        for slug, eng in engines.items():
            now = kpi_sum(eng); was = base_sum[slug] or 1e-9
            project_gain[slug] = {'было': round(was, 3), 'стало': round(now, 3),
                                  'прирост_%': round((now - was) / was * 100, 1)}
        for (slug, L), amt in alloc.items():
            result.append({'project': slug, 'node': L,
                           'name': str(engines[slug].G.nodes[L].get('name', L)),
                           'budget_added': round(amt, 2)})
        result.sort(key=lambda r: -r['budget_added'])
    finally:
        for slug, eng in engines.items():
            _restore(eng, snaps[slug])
    return {'allocations': result, 'project_gain': project_gain, 'spent': round(spent, 2)}


class ProjectMixer:
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, llm_engine: LocalLLMEngine, methodologies: Optional[Dict]=None, unmatched_finances: list=None, config: Optional[MixerConfig]=None, use_cached_weights: bool=False, weights_path: str="weights_matrix.json", progress_callback=None):
        self.unmatched_finances = unmatched_finances or [] # <--- НОВОЕ
        self.config = config or MixerConfig()
        self.llm_engine = llm_engine
        self.methodologies = methodologies or {}
        self.use_cached_weights = use_cached_weights
        self.weights_path = weights_path
        # Колбэк прогресса фоновой сборки (этап, деталь, доля 0..1). По умолчанию — заглушка.
        self._progress = progress_callback or (lambda *a, **k: None)
        # ДОБАВЛЕНО (#11): каталог пофайловых весов по каждому KPI рядом с общим файлом.
        self.weights_dir = os.path.join(os.path.dirname(os.path.abspath(weights_path)) or ".", "weights")
        # РЕАЛИЗОВАНО: пофайловая калибровка точности по каждому KPI (рядом с кешем весов).
        self.calibration_path = os.path.join(os.path.dirname(self.weights_dir) or ".", "calibration.json")
        self.kpi_calibration: Dict[str, Dict[str, Any]] = {}
        self.precedence: Dict[str, set] = {}   # зависимости предшествования (finish-to-start): pred → {succ}
        self.G = nx.DiGraph()
        self.kpi_ids = []
        self.kpi_weights: Dict[str, Dict[Tuple[str,str], Dict]] = {}
        self.budget_discrepancies = {}
        self.schedule_violations = {}      # ДОБАВЛЕНО: окна подзадач вне окна родителя
        self.quarter_windows = []          # ДОБАВЛЕНО: 4 временные фазы проекта
        self.horizon = (None, None)        # ДОБАВЛЕНО: общий горизонт проекта

        self._progress("Построение графа задач и показателей", frac=0.05)
        self._build_graph(nodes_df, edges_df)
        self._assert_dag()                 # ДОБАВЛЕНО: граф обязан быть ациклическим
        self._compute_quarter_windows()    # ДОБАВЛЕНО: фазы для поквартальной проекции
        self._build_kpi_weights(edges_df, nodes_df)
        self._load_calibration()           # калибровка точности по KPI (sensitivity/confidence/driver)
        self._build_kpi_calibration()      # предложения LLM по методике для отсутствующих
        self._progress("Расчёт базовых значений", frac=0.95)
        self._initial_calculate_all()
        self._validate_budgets()
        self._validate_schedule()          # ДОБАВЛЕНО
        self._progress("Готово", frac=1.0)

    @staticmethod
    def _pdate(s):
        """Безопасный разбор даты 'YYYY-MM-DD' (или datetime) в datetime."""
        if isinstance(s, datetime):
            return s
        return datetime.strptime(str(s)[:10], "%Y-%m-%d")

    def _assert_dag(self):
        """Гарантия ацикличности: топологическая сортировка и проекция по уровням
        корректны только на DAG. Раньше цикл во входных данных приводил к
        невнятному исключению глубоко внутри расчёта."""
        if not nx.is_directed_acyclic_graph(self.G):
            try:
                cycle = nx.find_cycle(self.G)
            except Exception:
                cycle = []
            raise ValueError(f"Граф проекта содержит цикл и не может быть рассчитан: {cycle}")

    def _compute_quarter_windows(self, n: int = 4):
        """Делит горизонт проекта [min(T_start), max(T_end)] на n равных фаз.

        Четыре плановых периода KPI (I–IV) трактуются как четыре равные временные
        фазы жизненного цикла проекта. Это позволяет детерминированно проецировать
        окно любой работы на отчётные периоды независимо от календарных кварталов."""
        starts, ends = [], []
        for _, a in self.G.nodes(data=True):
            if str(a.get('type', '')).upper() == 'KPI':
                continue
            try:
                starts.append(self._pdate(a.get('T_start')))
                ends.append(self._pdate(a.get('T_end')))
            except Exception:
                continue
        if not starts or not ends:
            self.quarter_windows, self.horizon = [], (None, None)
            return
        g0, g1 = min(starts), max(ends)
        self.horizon = (g0, g1)
        total = max(1, (g1 - g0).days)
        self.quarter_windows = []
        for i in range(n):
            a = g0 + timedelta(days=total * i // n)
            b = g0 + timedelta(days=total * (i + 1) // n)
            self.quarter_windows.append((a, b))

    def _waterfall_distribute(self, children, weights: Dict[str, float], old_F: Dict[str, float], delta_F: float) -> Dict[str, float]:
        """«Водопад»: распределяет ИЗМЕНЕНИЕ бюджета Δ между детьми по весу влияния,
        СОХРАНЯЯ суммарный бюджет (Σ детей = Σ старых + Δ). Бюджет не уходит в минус.

        При урезании (Δ<0): если требуемый срез для подзадачи превышает её запас, она
        обнуляется, а нераспределённый остаток дефицита каскадно переносится на оставшиеся
        активные работы пропорционально их весу. Раньше отрицательный бюджет просто
        обрезался max(0,…) — часть среза «сгорала», и сумма детей переставала совпадать с
        родителем (ложные срабатывания валидатора). Теперь срез применяется полностью."""
        new_F = {c: float(old_F.get(c, 0.0)) for c in children}
        if not children or abs(delta_F) < 1e-12:
            return new_F
        if delta_F >= 0:  # увеличение — просто по весу, без обрезаний
            total_w = sum(weights.get(c, 0.0) for c in children) or float(len(children))
            for c in children:
                w = weights.get(c, 0.0) if total_w else 1.0
                new_F[c] += delta_F * (w / total_w)
            return new_F
        # урезание: каскад
        to_remove = -delta_F
        active = [c for c in children if new_F[c] > 1e-12]
        guard = 0
        while to_remove > 1e-9 and active and guard <= len(children) + 2:
            guard += 1
            total_w = sum(weights.get(c, 0.0) for c in active)
            use_equal = total_w <= 0
            if use_equal:
                total_w = float(len(active))
            removed = 0.0
            still = []
            for c in active:
                w = 1.0 if use_equal else weights.get(c, 0.0)
                cut = to_remove * (w / total_w)
                if cut >= new_F[c]:           # запас исчерпан — обнуляем, остаток уйдёт дальше
                    removed += new_F[c]
                    new_F[c] = 0.0
                else:
                    new_F[c] -= cut
                    removed += cut
                    still.append(c)
            to_remove -= removed
            active = still
        return new_F

    @staticmethod
    def _is_milestone_type(t) -> bool:
        return str(t or '').strip().lower() in ('веха', 'milestone', 'веха.')

    def _ancestor_ref_days(self, node_id: str) -> float:
        """Масштаб времени для вехи — плановая длительность ближайшего длительного предка
        (мероприятия), внутри которого она является ключевой точкой. Запасной масштаб — квартал."""
        parts = [p for p in str(node_id).split('.') if p]
        while len(parts) > 1:
            parts = parts[:-1]
            cur = ".".join(parts)
            if cur in self.G.nodes:
                a = self.G.nodes[cur]
                try:
                    d = (self._pdate(a.get('T_end')) - self._pdate(a.get('T_start'))).days
                except Exception:
                    d = 0
                if d and d > 0:
                    return float(max(30, d))
        return 90.0  # квартал по умолчанию

    def _milestone_time_term(self, node_id: str, actual_end) -> float:
        """Временной член вехи: штраф за ОПОЗДАНИЕ даты достижения относительно ПЛАНОВОЙ.

        У вехи нет длительности, поэтому отклонение считается не от длительности, а от
        отклонения даты достижения: delay = (факт_конца − план_конца) в днях, нормированное
        длительностью родительского мероприятия. В срок → 0.5·β; позже → меньше; раньше →
        больше (если включён бонус за опережение)."""
        a = self.G.nodes[node_id]
        plan_end = a.get('T_plan_end') or a.get('T_end')
        try:
            delay = (self._pdate(actual_end) - self._pdate(plan_end)).days
        except Exception:
            delay = 0
        if (not self.config.time_bonus_enabled) and delay < 0:
            delay = 0  # опережение без бонуса не повышает ценность
        ref = self._ancestor_ref_days(node_id)
        return self.config.beta * stable_sigmoid(-self.config.sigmoid_k * (float(delay) / ref))

    def _milestone_value(self, node_id: str, F_real: float, actual_end) -> float:
        """Ценность вехи = бюджетный член + штраф за опоздание даты достижения."""
        val_F = self.config.alpha * np.log1p(self._eff_lambda() * max(0.0, float(F_real)))
        return float(val_F + self._milestone_time_term(node_id, actual_end))

    def _default_calib(self) -> Dict[str, Any]:
        # Нейтрально: sensitivity=1, confidence=1 → m не меняется (текущее поведение).
        # link_type — связь плана с KPI (прямой/косвенный); kpi_type — тип по топологии;
        # value_max — потолок значения (например, 100 для долей/процентов; None — без потолка).
        return {'sensitivity': 1.0, 'confidence': 1.0, 'driver': 'оба',
                'link_type': '—', 'kpi_type': '', 'formula': '', 'value_max': None, 'rationale': '',
                'agg_mode': getattr(self.config, 'default_agg_mode', 'ces'),
                'ces_rho': getattr(self.config, 'default_ces_rho', 1.0)}

    def classify_kpi_type(self, kpi_id: str) -> str:
        """Тип показателя по ТОПОЛОГИИ графа:
        • 'показатель=задача' — единственная листовая задача в поддереве;
        • 'общий показатель'  — поддерево покрывает почти все задачи проекта;
        • 'несколько задач'   — промежуточный случай."""
        sub = nx.ancestors(self.G, kpi_id)
        leaves_k = [n for n in sub if self.G.in_degree(n) == 0
                    and str(self.G.nodes[n].get('type', '')).upper() != 'KPI']
        all_leaves = [n for n in self.G.nodes if self.G.in_degree(n) == 0
                      and str(self.G.nodes[n].get('type', '')).upper() != 'KPI']
        nk, na = len(leaves_k), len(all_leaves)
        if nk <= 1:
            return 'показатель=задача'
        if na > 1 and nk >= 0.75 * na:
            return 'общий показатель'
        return 'несколько задач'

    def _type_default_calib(self, ktype: str) -> Dict[str, Any]:
        """Дефолтная калибровка по типу KPI (когда нет предложения LLM): тип 3 — высокая
        атрибуция (точно = задача), общий — низкая (вклад задачи мал), несколько — средняя."""
        if ktype == 'показатель=задача':
            return {'sensitivity': 1.0, 'confidence': 1.0,
                    'rationale': 'показатель = одна задача: прямая, точная атрибуция'}
        if ktype == 'общий показатель':
            return {'sensitivity': 1.0, 'confidence': 0.6,
                    'rationale': 'общий показатель: вклад отдельной задачи мал, атрибуция слабая'}
        return {'sensitivity': 1.0, 'confidence': 0.85,
                'rationale': 'несколько задач: средняя атрибуция'}

    def get_kpi_calibration(self, kpi_id: str) -> Dict[str, Any]:
        return {**self._default_calib(), **self.kpi_calibration.get(kpi_id, {})}

    def _clamp_kpi_forecast(self, kpi_id: str, val: float) -> float:
        """Границы значения KPI: пол 0 и потолок value_max (например, 100% для долей)."""
        v = max(0.0, float(val))
        vmax = self.get_kpi_calibration(kpi_id).get('value_max')
        if vmax is not None:
            try:
                v = min(float(vmax), v)
            except Exception:
                pass
        return v

    def _calibrate_m(self, kpi_id: str, m_raw: float) -> float:
        """Заземление прогноза без исторических данных:
        m = 1 + confidence · sensitivity · (m_raw − 1).
        • sensitivity масштабирует силу реакции KPI (грубая «эластичность» из методики/эксперта);
        • confidence < 1 «притягивает» прогноз к плану там, где доверия меньше (снижает ожидаемую
          ошибку при неопределённости — регуляризация к единственной достоверной точке, плану)."""
        c = self.get_kpi_calibration(kpi_id)
        s = float(c.get('sensitivity', 1.0)); conf = float(c.get('confidence', 1.0))
        return max(0.0, 1.0 + conf * s * (float(m_raw) - 1.0))

    def set_precedence(self, pred: str, succ: str, on: bool = True):
        """Зависимость предшествования finish-to-start: succ не может начаться раньше конца pred."""
        if pred not in self.G.nodes or succ not in self.G.nodes or pred == succ:
            return
        if on:
            self.precedence.setdefault(pred, set()).add(succ)
        else:
            self.precedence.get(pred, set()).discard(succ)

    def get_precedence_edges(self) -> List[Tuple[str, str]]:
        return [(p, s) for p, ss in self.precedence.items() for s in ss]

    def _precedence_cascade(self, start_task: str) -> List[str]:
        """Каскад срыва срока: если из-за переноса конца pred его последователь начинался бы
        раньше — двигаем последователя (сохраняя длительность), и так далее по цепочке.
        Возвращает список сдвинутых работ. Веха остаётся точкой."""
        if not self.precedence:
            return []
        shifted, seen, queue = [], set(), [start_task]
        while queue:
            p = queue.pop(0)
            if p in seen:
                continue
            seen.add(p)
            try:
                p_end = self._pdate(self.G.nodes[p].get('T_end'))
            except Exception:
                continue
            for s in list(self.precedence.get(p, ())):
                sa = self.G.nodes.get(s)
                if not sa:
                    continue
                try:
                    s_start = self._pdate(sa.get('T_start')); s_end = self._pdate(sa.get('T_end'))
                except Exception:
                    continue
                if s_start < p_end:  # нарушение — двигаем последователя к концу предшественника
                    dur = max(0, (s_end - s_start).days)
                    is_ms = self._is_milestone_type(sa.get('type'))
                    new_start = p_end
                    new_end = p_end + timedelta(days=dur)
                    if is_ms:
                        new_start = new_end
                    sa['T_start'] = new_start.strftime('%Y-%m-%d')
                    sa['T_end'] = new_end.strftime('%Y-%m-%d')
                    if self.G.in_degree(s) == 0:  # лист — пересчитать ценность
                        if is_ms:
                            sa['local_value'] = self._milestone_value(s, float(sa.get('F', 0.0)), sa['T_end'])
                        else:
                            d = max(1, (new_end - new_start).days)
                            sa['local_value'] = self._calculate_local_value(float(sa.get('F', 0.0)), d, T_opt=sa.get('T_opt', d), late_days=self._late_days(sa, sa.get('T_end')))
                    shifted.append(s)
                    queue.append(s)
        return shifted

    def set_kpi_calibration(self, kpi_id, sensitivity=None, confidence=None, driver=None,
                            rationale=None, link_type=None, formula=None, value_max=None, kpi_type=None,
                            agg_mode=None, ces_rho=None):
        c = dict(self.kpi_calibration.get(kpi_id, self._default_calib()))
        if sensitivity is not None:
            c['sensitivity'] = float(np.clip(sensitivity, 0.0, 3.0))
        if confidence is not None:
            c['confidence'] = float(np.clip(confidence, 0.0, 1.0))
        if driver is not None:
            c['driver'] = str(driver)
        if link_type is not None:
            c['link_type'] = str(link_type)
        if formula is not None:
            c['formula'] = str(formula)[:400]
        if kpi_type is not None:
            c['kpi_type'] = str(kpi_type)
        if value_max is not None:
            c['value_max'] = (None if value_max == '' else float(value_max))
        if agg_mode is not None:
            c['agg_mode'] = ('ces' if str(agg_mode).lower() == 'ces' else 'классический')
            c['agg_source'] = 'user'  # пользователь задал режим вручную → дефолт его не перезапишет
        if ces_rho is not None:
            c['ces_rho'] = float(np.clip(ces_rho, -8.0, 4.0))
            c['agg_source'] = 'user'
        if rationale is not None:
            c['rationale'] = str(rationale)[:400]
        self.kpi_calibration[kpi_id] = c
        self._save_calibration()

    def shapley_attribution(self, kpi_id: str, max_players: Optional[int] = None, samples: int = 120, seed: int = 0) -> List[Dict[str, Any]]:
        """Корректное разложение ценности KPI по работам методом ШЕПЛИ (Монте-Карло).

        В отличие от «влияния» (которое не суммируется в 100%, т.к. меряет удаление по одному),
        вклады Шепли учитывают взаимодействия и СУММИРУЮТСЯ в полную ценность KPI — это честный
        «водопад вкладов работ» и ответ на вопрос «из чего складывается показатель».
        По умолчанию (max_players=None) считаются ВСЕ питающие работы по отдельности; если задать
        max_players, дорогих участников оставляем по отдельности, а остальных группируем в «прочие»."""
        sub = nx.ancestors(self.G, kpi_id)
        leaves = [n for n in sub if self.G.in_degree(n) == 0
                  and str(self.G.nodes[n].get('type', '')).upper() != 'KPI']
        if not leaves:
            return []
        # игроки: по умолчанию — ВСЕ работы; при заданном лимите топ по влиянию + групповой «прочие»
        infl = sorted(leaves, key=lambda n: self.kpi_influence(n, kpi_id), reverse=True)
        if max_players is not None and len(infl) > max_players:
            top = infl[:max_players]
            rest = infl[max_players:]
            players = {n: {n} for n in top}
            if rest:
                players['__прочие__'] = set(rest)
        else:
            players = {n: {n} for n in infl}  # все работы по отдельности
        pids = list(players.keys())

        snap = {l: self.G.nodes[l].get('local_value', 0.0) for l in leaves}

        def v_of(active_leaves: set) -> float:
            for l in leaves:
                self.G.nodes[l]['local_value'] = snap[l] if l in active_leaves else 0.0
            self._propagate_single_kpi(kpi_id)
            return self._kpi_value(kpi_id)

        try:
            rng = random.Random(seed)
            phi = {p: 0.0 for p in pids}
            for _ in range(max(1, samples)):
                perm = pids[:]
                rng.shuffle(perm)
                active = set()
                prev = 0.0  # v(∅): все листья обнулены
                for p in perm:
                    active |= players[p]
                    cur = v_of(active)
                    phi[p] += (cur - prev)
                    prev = cur
            for p in pids:
                phi[p] /= max(1, samples)
        finally:
            for l, lv in snap.items():
                self.G.nodes[l]['local_value'] = lv
            self._propagate_single_kpi(kpi_id)

        total = sum(phi.values()) or 1.0
        rows = []
        for p in pids:
            name = 'Прочие работы' if p == '__прочие__' else str(self.G.nodes[p].get('name', p))
            rows.append({'node': p, 'Работа': (p if p != '__прочие__' else '—') + (f" — {name}" if p != '__прочие__' else name),
                         'Вклад': round(phi[p], 4), 'Доля': round(phi[p] / total, 4)})
        rows.sort(key=lambda r: r['Вклад'], reverse=True)
        return rows

    def iter_calibration_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for kpi in self.kpi_ids:
            c = self.get_kpi_calibration(kpi)
            vmax = c.get('value_max')
            rows.append({
                'kpi_id': kpi, 'KPI': str(self.G.nodes[kpi].get('name', kpi)),
                'Тип KPI': c.get('kpi_type', '') or self.classify_kpi_type(kpi),
                'Чувствительность': round(float(c['sensitivity']), 2),
                'Уверенность': round(float(c['confidence']), 2),
                'Драйвер': c.get('driver', 'оба'),
                'Связь': c.get('link_type', '—'),
                'Потолок': ('' if vmax is None else round(float(vmax), 2)),
                'Режим': c.get('agg_mode', 'классический'),
                'ρ (CES)': round(float(c.get('ces_rho', 1.0)), 2),
                'Формула': c.get('formula', ''),
                'Обоснование': c.get('rationale', ''),
            })
        return rows

    def _load_calibration(self):
        self.kpi_calibration = {}
        try:
            if os.path.exists(self.calibration_path):
                with open(self.calibration_path, 'r', encoding='utf-8') as f:
                    self.kpi_calibration = {str(k): v for k, v in (json.load(f) or {}).items()}
        except Exception as e:
            logger.warning(f"Не удалось загрузить калибровку KPI: {e}")
            self.kpi_calibration = {}

    def _save_calibration(self):
        try:
            _atomic_write_text(self.calibration_path,
                               json.dumps(self.kpi_calibration, ensure_ascii=False, indent=1))
        except Exception as e:
            logger.warning(f"Не удалось сохранить калибровку KPI: {e}")

    def _build_kpi_calibration(self):
        """Калибровка по каждому KPI. Для отсутствующих: предложение LLM по методике, иначе —
        ДЕФОЛТ ПО ТИПУ KPI (тип определяется топологией). Всегда проставляются тип и потолок
        значения (100 для долей/процентов). Кешируется."""
        changed = False
        for idx, kpi in enumerate(self.kpi_ids):
            ktype = self.classify_kpi_type(kpi)
            name = str(self.G.nodes[kpi].get('name', kpi))
            is_pct = any(s in name.lower() for s in ('%', 'процент', 'доля'))
            if kpi in self.kpi_calibration:
                # дозаполняем тип/потолок для ранее сохранённых записей
                ent = self.kpi_calibration[kpi]
                if not ent.get('kpi_type'):
                    ent['kpi_type'] = ktype; changed = True
                if 'value_max' not in ent:
                    ent['value_max'] = (100.0 if is_pct else None); changed = True
                # Режим агрегации/ρ из ПАНЕЛИ (дефолт) применяем к KPI, которые пользователь
                # не настраивал вручную (agg_source != 'user'). Иначе смена режима в панели
                # не действовала бы: кешированная калибровка перекрывала бы новый дефолт.
                if ent.get('agg_source') != 'user':
                    def_mode = getattr(self.config, 'default_agg_mode', 'ces')
                    def_rho = getattr(self.config, 'default_ces_rho', 1.0)
                    if ent.get('agg_mode') != def_mode or float(ent.get('ces_rho', 1.0)) != float(def_rho):
                        ent['agg_mode'] = def_mode
                        ent['ces_rho'] = def_rho
                        changed = True
                continue
            self._progress(f"Калибровка KPI {idx + 1}/{len(self.kpi_ids)}", frac=0.93)
            meth = self.methodologies.get(kpi) or self.methodologies.get(name, '')
            prop = None
            try:
                if meth and hasattr(self.llm_engine, 'propose_kpi_calibration'):
                    prop = self.llm_engine.propose_kpi_calibration(name, meth)
            except Exception as e:
                logger.warning(f"Калибровка KPI {kpi}: предложение LLM не получено ({e})")
                prop = None
            base = self._type_default_calib(ktype) if not prop else {}
            self.kpi_calibration[kpi] = {
                **self._default_calib(), **base, **(prop or {}),
                'kpi_type': ktype,
                'value_max': (100.0 if is_pct else None),
            }
            changed = True
        if changed:
            self._save_calibration()

    # Эффективный масштаб бюджета: lambda_f задан «на миллион»; для других единиц
    # пересчитываем (через BUDGET_UNITS), чтобы «колено» отдачи оставалось около типичного бюджета.
    def _eff_lambda(self) -> float:
        # Единый источник масштаба — BUDGET_UNITS (без дубля коэффициентов).
        unit = BUDGET_UNITS.get(getattr(self.config, 'budget_scale', 'millions'), BUDGET_UNITS['millions'])
        return self.config.lambda_f * unit['factor']
    
    # --- НОВОЕ: Шаг 2 и 3 (Оценка профиля финансов) ---
    def _evaluate_node_finances(self, finances, rho_req: float = 1.0, rho_add: float = 0.0) -> Tuple[float, float, Optional[int]]:
        """Возвращает (номинальный эффективный бюджет, реальный дисконтированный бюджет, год закрытия финансирования)."""
        if isinstance(finances, str):
            try: 
                finances = json.loads(finances)
                if isinstance(finances, str): 
                    finances = json.loads(finances)
            except Exception: 
                finances = {}
        if not isinstance(finances, dict) or not finances:
            return 0.0, 0.0, None
            
        f_eff = 0.0
        f_real = 0.0
        last_year = None
        rate = getattr(self.config, 'discount_rate', 0.06)
        base_y = getattr(self.config, 'base_year', 2026)
        
        for y_str, amounts in finances.items():
            try: y = int(y_str)
            except Exception: continue
            
            # Взвешенный бюджет с локальными коэффициентами задачи
            val = float(amounts.get('base', 0.0)) + float(amounts.get('req_extra', 0.0)) * rho_req + float(amounts.get('add', 0.0)) * rho_add
            if val > 1e-9:
                if last_year is None or y > last_year:
                    last_year = y
                f_eff += val
                f_real += val / ((1.0 + rate) ** max(0, y - base_y))
                
        return f_eff, f_real, last_year

    def _late_days(self, attrs: Dict[str, Any], actual_end) -> int:
        """Сколько дней работа завершается ПОЗЖЕ планового конца (T_plan_end). 0 — в срок/раньше.
        Если штраф за позднее завершение выключен — всегда 0 (поведение не меняется)."""
        if not getattr(self.config, 'late_finish_penalty_enabled', False):
            return 0
        plan_end = attrs.get('T_plan_end') or attrs.get('T_end')
        try:
            return max(0, (self._pdate(actual_end) - self._pdate(plan_end)).days)
        except Exception:
            return 0

    def _calculate_local_value(self, F_real: float, delta_days: int, T_opt: int = 30,
                               late_days: int = 0) -> float:
        """Ценность ЛИСТА-РАБОТЫ (Задача/Подзадача/Мероприятие): бюджет + отклонение ДЛИТЕЛЬНОСТИ.
        Для ВЕХ используется отдельная функция _milestone_value (бюджет + опоздание ДАТЫ) — это
        единственный источник ценности вехи."""
        val_F = self.config.alpha * np.log1p(self._eff_lambda() * max(0.0, float(F_real)))
        T_opt = max(1, int(T_opt))  # защита от деления на ноль
        if not self.config.time_bonus_enabled and delta_days < T_opt:
            delta_days = T_opt
        # ИСПРАВЛЕНО (аудит): временной член считается от ОТНОСИТЕЛЬНОГО отклонения срока
        # (Δ/T_opt − 1), а не от разницы в днях. Так sigmoid_k безразмерен и не зависит от
        # масштаба длительностей, нет скачка значения от сдвига на 1 день, нет переполнения.
        rel_dev = (float(delta_days) / float(T_opt)) - 1.0
        val_T = self.config.beta * stable_sigmoid(-self.config.sigmoid_k * rel_dev)
        val = val_F + val_T
        # ДОБАВЛЕНО (3): отдельный штраф за ПОЗДНЕЕ ЗАВЕРШЕНИЕ относительно плановой даты конца.
        # Раньше наказывалось только отклонение ДЛИТЕЛЬНОСТИ от оптимума; теперь сдвиг работы
        # «вправо» (даже без изменения длительности) тоже ухудшает ценность — как у вех.
        if late_days and late_days > 0:
            x = float(late_days) / float(T_opt)
            late_pen = self.config.late_finish_weight * self.config.beta * (2.0 * stable_sigmoid(self.config.sigmoid_k * x) - 1.0)
            val -= late_pen
        return float(max(0.0, val))

    def _build_graph(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        target_ids = set(edges_df['target'].tolist()) if not edges_df.empty else set()

        for _, row in nodes_df.iterrows():
            n_id = row['id']
            n_type = str(row['type']).upper()
            row_dict = row.to_dict()
            if 'finances' in row_dict and isinstance(row_dict['finances'], str):
                try:
                    row_dict['finances'] = json.loads(row_dict['finances'])
                except Exception:
                    row_dict['finances'] = {}
            if n_type == 'KPI':
                self.G.add_node(n_id, **row.to_dict(), local_value=0.0, agg_value=0.0, agg_by_kpi={})
                self.kpi_ids.append(n_id)
                continue
            if n_id not in target_ids:
                is_ms = self._is_milestone_type(row.get('type'))
                if is_ms:
                    # Ценность вехи выставит пост-проход через _milestone_value (нужны рёбра/родитель).
                    l_val = 0.0
                else:
                    delta_days = max(1, (pd.to_datetime(row['T_end']) - pd.to_datetime(row['T_start'])).days)
                    l_val = self._calculate_local_value(row['F'], delta_days, T_opt=row['T_opt'])
            else:
                l_val = 0.0
            self.G.add_node(n_id, **row.to_dict(), local_value=l_val, agg_value=0.0, agg_by_kpi={})
            # ПЛАНОВАЯ дата конца фиксируется при сборке — относительно неё считается опоздание
            # (для штрафа за позднее завершение и для вех). Меняется только при утверждении плана.
            if n_id not in target_ids and n_type != 'KPI':
                self.G.nodes[n_id].setdefault('T_plan_end', row.get('T_end'))

        for _, row in edges_df.iterrows():
            if row['source'] in self.G and row['target'] in self.G:
                self.G.add_edge(row['source'], row['target'], weight=1.0, relation_type='linear')

        # Пост-проход: у листовых ВЕХ запоминаем ПЛАНОВУЮ дату достижения (T_plan_end) и
        # пересчитываем ценность с временным членом — штрафом за опоздание относительно плана
        # (ближайшие предки/мероприятия уже в графе, масштаб опоздания известен).
        for nid in self.G.nodes():
            a = self.G.nodes[nid]
            if str(a.get('type', '')).upper() == 'KPI':
                continue
            if self.G.in_degree(nid) == 0 and self._is_milestone_type(a.get('type')):
                a.setdefault('T_plan_end', a.get('T_end'))
                a['local_value'] = self._milestone_value(nid, a.get('F', 0.0), a.get('T_end'))

        # РЕЖИМ «ПО ДОЛЕ ВЫПОЛНЕНИЯ ПЛАНА» (вариант Б): фиксируем ПЛАНОВУЮ ценность листа —
        # его ценность при плановых бюджете/сроках. При сборке всё на плане, поэтому плановая
        # ценность = текущей. Доля выполнения = текущая ценность / плановая (1.0 = ровно по плану).
        for nid in self.G.nodes():
            a = self.G.nodes[nid]
            if str(a.get('type', '')).upper() != 'KPI' and self.G.in_degree(nid) == 0:
                a['v_plan'] = float(a.get('local_value', 0.0))

        self._build_propagation_cache()

    def _build_propagation_cache(self):
        """ОПТИМИЗАЦИЯ: топологический порядок графа и порядок обхода подграфа каждого KPI
        зависят только от СТРУКТУРЫ (она фиксируется при сборке). Считаем их ОДИН раз и
        переиспользуем — иначе каждый propagate заново звал nx.topological_sort/nx.ancestors
        (в горячих циклах influence/Шепли/портфель это были тысячи лишних обходов)."""
        topo = list(nx.topological_sort(self.G))
        self._topo_order = topo
        self._kpi_sub_order = {}
        for kpi_id in self.kpi_ids:
            anc = nx.ancestors(self.G, kpi_id) | {kpi_id}
            self._kpi_sub_order[kpi_id] = [n for n in topo if n in anc]

    def _kpi_weight_signature(self, kpi_id: str, sub_edges: pd.DataFrame, nodes_df: pd.DataFrame) -> str:
        """Структурная подпись подграфа KPI (#11).

        Меняется ТОЛЬКО при обновлении сущностей и их свойств из план-графика, относящихся
        к этому KPI: состав рёбер, типы/имена узлов и текст методики. НЕ зависит от текущего
        бюджета/дат (они меняются в ходе симуляции и не должны вызывать пересчёт весов)."""
        edges = sorted((str(r['source']), str(r['target'])) for _, r in sub_edges.iterrows())
        node_ids = sorted(set([s for s, _ in edges] + [t for _, t in edges] + [kpi_id]))
        meta = []
        ndf = nodes_df.set_index('id') if 'id' in nodes_df.columns else None
        for nid in node_ids:
            typ = name = ''
            if ndf is not None and nid in ndf.index:
                rec = ndf.loc[nid]
                typ = str(rec['type']) if 'type' in ndf.columns else ''
                name = str(rec['name']) if 'name' in ndf.columns else ''
            meta.append(f"{nid}|{typ}|{name}")
        meth = self.methodologies.get(kpi_id, "") or ""
        payload = json.dumps({'edges': edges, 'nodes': meta, 'meth': hashlib.md5(meth.encode('utf-8')).hexdigest()},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    def _kpi_weight_file(self, kpi_id: str) -> str:
        return os.path.join(self.weights_dir, f"{kpi_id}.json")

    def _save_kpi_weights_file(self, kpi_id: str, signature: str):
        """Пишет веса одного KPI в его файл (вызывается по мере расчёта — инкрементально)."""
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            wdict = self.kpi_weights.get(kpi_id, {})
            payload = {
                'kpi_id': kpi_id, 'signature': signature,
                'edges': [{'source': s, 'target': t, 'weight': d['weight'],
                           'relation_type': d.get('relation_type', 'linear'),
                           'rationale': d.get('rationale', '')}
                          for (s, t), d in wdict.items()],
            }
            _atomic_write_text(self._kpi_weight_file(kpi_id), json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"Не удалось сохранить веса KPI {kpi_id}: {e}")

    def _load_kpi_weights_file(self, kpi_id: str):
        """Читает веса одного KPI из его файла → (signature, weights_dict) или (None, None)."""
        path = self._kpi_weight_file(kpi_id)
        if not os.path.exists(path):
            return None, None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            d = {(e['source'], e['target']): {'weight': e['weight'],
                                              'relation_type': e.get('relation_type', 'linear'),
                                              'rationale': e.get('rationale', '')}
                 for e in data.get('edges', [])}
            return data.get('signature'), d
        except Exception as e:
            logger.warning(f"Не удалось прочитать веса KPI {kpi_id}: {e}")
            return None, None

    def _build_kpi_weights(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
        # ИЗМЕНЕНО (#11): веса хранятся ПОФАЙЛОВО по каждому KPI и пересчитываются только
        # при изменении структурной подписи подграфа. Каждый KPI сохраняется сразу после
        # расчёта (инкрементально), что не замедляет обычные прогоны и не теряет прогресс.
        if self.use_cached_weights:
            self._migrate_legacy_weights_file()  # перенос старого общего файла, если есть

        for idx, kpi_id in enumerate(self.kpi_ids):
            kpi_name = str(self.G.nodes[kpi_id].get('name', kpi_id))[:60]
            self._progress(f"Сжатие методик и расчёт весов: KPI {idx + 1}/{len(self.kpi_ids)}",
                           detail=kpi_name, frac=0.15 + 0.78 * (idx / max(1, len(self.kpi_ids))))
            ancestors = nx.ancestors(self.G, kpi_id)
            subgraph_nodes = ancestors | {kpi_id}
            sub_edges = edges_df[edges_df['source'].isin(subgraph_nodes) & edges_df['target'].isin(subgraph_nodes)]
            current_sig = self._graph_kpi_signature(kpi_id)

            if self.use_cached_weights:
                cached_sig, cached_w = self._load_kpi_weights_file(kpi_id)
                if cached_sig == current_sig and cached_w is not None:
                    self.kpi_weights[kpi_id] = cached_w
                    continue  # структура не менялась — берём из кеша, LLM не зовём
                if cached_sig is not None:
                    logger.info(f"Веса {kpi_id}: структура изменилась — пересчёт.")

            if self.llm_engine is not None:
                weights = self.llm_engine.extract_semantic_weights_for_kpi(
                    sub_edges, self.methodologies, nodes_df, kpi_id, subgraph_nodes
                )
            else:
                weights = {}
            for _, row in sub_edges.iterrows():
                src, tgt = str(row['source']), str(row['target'])
                if tgt == kpi_id and (src, tgt) not in weights:
                    weights[(src, tgt)] = {'weight': 1.0, 'relation_type': 'direct', 'rationale': 'прямая связь задача→KPI'}
            self.kpi_weights[kpi_id] = weights
            self._save_kpi_weights_file(kpi_id, current_sig)  # инкрементальная запись

    def _migrate_legacy_weights_file(self):
        """Однократно переносит старый общий weights_matrix.json в пофайловый формат (#11)."""
        if not os.path.exists(self.weights_path):
            return
        try:
            with open(self.weights_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            if not isinstance(cached_data, dict):
                return
            os.makedirs(self.weights_dir, exist_ok=True)
            for kpi_id, edge_list in cached_data.items():
                if os.path.exists(self._kpi_weight_file(kpi_id)):
                    continue
                d = {(e['source'], e['target']): {'weight': e['weight'],
                                                  'relation_type': e.get('relation_type', 'linear'),
                                                  'rationale': e.get('rationale', '')}
                     for e in edge_list}
                self.kpi_weights[kpi_id] = d
                # подпись неизвестна (структуру проверим при первом расчёте) → пустая
                self._save_kpi_weights_file(kpi_id, signature="legacy")
            # старый файл больше не нужен как источник истины — переименуем в .bak
            try:
                os.replace(self.weights_path, self.weights_path + ".bak")
            except Exception:
                pass
            logger.info("Старый общий файл весов перенесён в пофайловый формат.")
        except Exception as e:
            logger.warning(f"Миграция старого файла весов не удалась: {e}")

    def _validate_budgets(self):
        self.budget_discrepancies.clear()
        for node in self.G.nodes():
            # KPI-узлы не имеют собственного бюджета и в проверке игнорируются (п.9 методики).
            if str(self.G.nodes[node].get('type', '')).upper() == 'KPI':
                continue
            # ИСПРАВЛЕНО: дети — это ПРЕДШЕСТВЕННИКИ (рёбра идут ребёнок→родитель).
            # Было successors() — туда попадал РОДИТЕЛЬ узла, и проверка бюджета
            # сравнивала бюджет узла с бюджетом его родителя (бессмыслица).
            children = [c for c in self.G.predecessors(node) if not str(self.G.nodes[c].get('type', '')).upper().startswith('KPI')]
            if not children:
                continue
            total_child_F = sum(float(self.G.nodes[child].get('F', 0.0)) for child in children)
            parent_F = float(self.G.nodes[node].get('F', 0.0))
            if abs(total_child_F - parent_F) > 0.001:
                self.budget_discrepancies[node] = {
                    'parent_F': parent_F,
                    'children_sum': total_child_F,
                    'diff': parent_F - total_child_F,
                    'children': children
                }

    def _validate_schedule(self):
        """Календарная согласованность WBS: окно каждой подзадачи должно лежать
        внутри окна родителя. Нарушения (ребёнок начинается раньше или заканчивается
        позже родителя) собираются по родителю — аналогично расхождениям бюджета."""
        self.schedule_violations.clear()
        for node in self.G.nodes():
            if str(self.G.nodes[node].get('type', '')).upper() == 'KPI':
                continue
            try:
                ps, pe = self._pdate(self.G.nodes[node].get('T_start')), self._pdate(self.G.nodes[node].get('T_end'))
            except Exception:
                continue
            children = [c for c in self.G.predecessors(node)
                        if str(self.G.nodes[c].get('type', '')).upper() != 'KPI']
            for child in children:
                try:
                    cs, ce = self._pdate(self.G.nodes[child].get('T_start')), self._pdate(self.G.nodes[child].get('T_end'))
                except Exception:
                    continue
                start_before = max(0, (ps - cs).days)
                end_after = max(0, (ce - pe).days)
                if start_before > 0 or end_after > 0:
                    self.schedule_violations.setdefault(node, []).append({
                        'child': child,
                        'child_name': self.G.nodes[child].get('name', child),
                        'start_before_days': start_before,
                        'end_after_days': end_after,
                        'overrun_days': max(start_before, end_after),
                    })

    def _edge_contribution(self, relation_type, weight: float, v: float):
        """Вклад одного ребра «ребёнок→родитель» с учётом ТИПА зависимости.

        Возвращает (величина_вклада, is_inhibitory). Положительные типы суммируются и
        проходят через порог активации узла; тормозящий — вычитается из агрегата.
        Все типы монотонны по ценности ребёнка v (тормозящий — монотонно убывает)."""
        rt = canonical_relation(relation_type)
        w = float(weight); v = max(0.0, float(v))
        k = self.config.sigmoid_activation_k
        tau = self.config.activation_threshold
        if rt == 'inhibitory':
            return w * v, True                                   # −w·v (знак учтён снаружи)
        if rt == 'saturating':
            return w * (np.log1p(v) / np.log(2.0)), False        # убывающая отдача (=w·v при v=1)
        if rt == 'threshold':
            return w * v * stable_sigmoid(k * (v - tau)), False  # гейт ниже порога τ
        if rt == 'amplifying':
            t = max(0.1, tau)
            return w * v * (1.0 + float(np.tanh(v / t))), False  # синергия: множитель 1…2
        return w * v, False                                       # linear (и провенансные метки)

    @staticmethod
    def _ces_combine(pairs: List[Tuple[float, float]], rho: float) -> float:
        """CES-агрегатор: (Σ wᵢ·vᵢ^ρ)^(1/ρ) — один параметр ρ непрерывно проходит от
        совершенных ЗАМЕНИТЕЛЕЙ (ρ=1, обычная сумма) через Кобба-Дугласа (ρ→0, геометрическое)
        к жёсткой КОМПЛЕМЕНТАРНОСТИ / «слабому звену» (ρ→−∞, минимум). Унифицирует линейный/
        насыщающий/совместный типы одной ручкой."""
        if not pairs:
            return 0.0
        rho = float(rho)
        if abs(rho) < 1e-6:  # ρ→0: взвешенное геометрическое
            sw = sum(w for w, _ in pairs) or 1.0
            prod = 1.0
            for w, v in pairs:
                prod *= max(1e-9, float(v)) ** (w / sw)
            return float(prod)
        s = 0.0
        for w, v in pairs:
            vv = max(0.0, float(v))
            if rho < 0 and vv <= 1e-9:
                return 0.0  # слабое звено = ноль обнуляет комплементарную свёртку
            s += float(w) * (vv ** rho)
        if s <= 0:
            return 0.0
        return float(s ** (1.0 / rho))

    def _propagate_for_kpi(self, kpi_id: str, nodes_list: List[str]):
        weights = self.kpi_weights.get(kpi_id, {})
        calib = self.get_kpi_calibration(kpi_id)
        use_ces = str(calib.get('agg_mode', '')).lower() == 'ces'
        rho = float(calib.get('ces_rho', 1.0))
        for node in nodes_list:
            local = self.G.nodes[node].get('local_value', 0.0)
            preds = list(self.G.predecessors(node))
            if preds:
                pos_raw = 0.0   # сумма положительных вкладов (с учётом типа) — классический режим
                inh_raw = 0.0   # сумма тормозящих вкладов
                pos_count = 0   # число фактических положительных вкладчиков
                pos_pairs = []  # (вес, ценность) для CES-режима
                for p in preds:
                    wd = weights.get((p, node), {})
                    w = wd.get('weight', 0.0)
                    rt = wd.get('relation_type', 'linear')
                    v = self.G.nodes[p].get('agg_by_kpi', {}).get(kpi_id, 0.0)
                    contrib, is_inhib = self._edge_contribution(rt, w, v)
                    if is_inhib:
                        inh_raw += contrib
                    else:
                        pos_raw += contrib
                        pos_pairs.append((w, v))
                        if contrib > 0:
                            pos_count += 1
                if use_ces:
                    # CES сам задаёт форму свёртки — порог активации и типы не применяются.
                    agg_val = max(0.0, local + self._ces_combine(pos_pairs, rho) - inh_raw)
                else:
                    threshold = self.config.activation_threshold
                    k = self.config.sigmoid_activation_k
                    # «Шумовой порог» нужен только при НЕСКОЛЬКИХ вкладах; единственный — без искажения.
                    activation = 1.0 if pos_count <= 1 else stable_sigmoid(k * (pos_raw - threshold))
                    agg_val = max(0.0, local + pos_raw * activation - inh_raw)
            else:
                # Лист: его агрегат = собственная ценность. В режиме «по доле выполнения плана»
                # вместо абсолютной ценности используется ДОЛЯ от плановой (1.0 = ровно по плану),
                # которая затем взвешенно (через веса = влияние) сворачивается выше.
                if str(getattr(self.config, 'forecast_mode', 'value')) == 'completion':
                    vp = float(self.G.nodes[node].get('v_plan', 0.0) or 0.0)
                    agg_val = min(3.0, max(0.0, local / vp)) if vp > 1e-9 else 1.0
                else:
                    agg_val = local
            self.G.nodes[node].setdefault('agg_by_kpi', {})[kpi_id] = agg_val

    def _sync_agg_value(self):
        """Сводит per-KPI значения к одному представительному agg_value для отображения.

        Для KPI-узла это его собственная агрегированная ценность; для обычной задачи —
        максимум по всем KPI, которые она питает (пиковая значимость). Полная разбивка
        по показателям доступна через get_node_kpi_values()."""
        for n in self.G.nodes():
            ab = self.G.nodes[n].get('agg_by_kpi', {})
            if str(self.G.nodes[n].get('type', '')).upper() == 'KPI':
                self.G.nodes[n]['agg_value'] = float(ab.get(n, 0.0))
            elif ab:
                self.G.nodes[n]['agg_value'] = float(max(ab.values()))
            else:
                self.G.nodes[n]['agg_value'] = float(self.G.nodes[n].get('local_value', 0.0))

    def _kpi_value(self, kpi_id: str) -> float:
        """Текущая агрегированная ценность конкретного KPI (под его собственными весами)."""
        return float(self.G.nodes[kpi_id].get('agg_by_kpi', {}).get(kpi_id, self.G.nodes[kpi_id].get('agg_value', 0.0)))

    def get_node_kpi_values(self, node: str) -> Dict[str, float]:
        """Разбивка ценности узла по каждому KPI, на который он влияет: {kpi_id: значение}."""
        return {k: float(v) for k, v in self.G.nodes[node].get('agg_by_kpi', {}).items()}

    def get_node_kpis(self, node: str) -> List[str]:
        """Список KPI (id), которые питает данный узел."""
        return list(self.G.nodes[node].get('agg_by_kpi', {}).keys())

    def task_kpi_influences(self, node: str) -> List[Dict[str, Any]]:
        """«Взгляд от задачи»: на какие KPI влияет узел и насколько (влияние = доля просадки
        KPI при обнулении узла). ВНИМАНИЕ: влияния на РАЗНЫЕ KPI несопоставимы и НЕ
        суммируются — у них разные знаменатели."""
        out = []
        for kpi in self.get_node_kpis(node):
            if kpi == node:
                continue
            if str(self.G.nodes.get(kpi, {}).get('type', '')).upper() != 'KPI':
                continue
            out.append({
                'kpi_id': kpi,
                'KPI': str(self.G.nodes[kpi].get('name', kpi)),
                'Влияние': round(self.kpi_influence(node, kpi), 4),
            })
        out.sort(key=lambda r: r['Влияние'], reverse=True)
        return out

    def get_node_kpi_names(self, node: str) -> List[str]:
        """Имена KPI, которые питает данный узел."""
        return [str(self.G.nodes[k].get('name', k)) for k in self.get_node_kpis(node) if k in self.G.nodes]

    def _child_distribution_weights(self, parent: str, children: List[str]) -> Dict[str, float]:
        """Доли распределения бюджета родителя между детьми (#M).

        По просьбе: бюджет распределяется ПРОПОРЦИОНАЛЬНО ВЕСУ влияния ребёнка
        (усреднённому по KPI, которые питает родитель), а не пропорционально текущему
        бюджету ребёнка. Если весов нет — делим поровну."""
        kpis = self.get_node_kpis(parent) or list(self.kpi_ids)
        raw = {}
        for c in children:
            ws = []
            for kpi in kpis:
                w = self.kpi_weights.get(kpi, {}).get((c, parent), {}).get('weight')
                if w is not None:
                    ws.append(float(w))
            raw[c] = (sum(ws) / len(ws)) if ws else 0.0
        total = sum(raw.values())
        if total <= 0:
            n = len(children) or 1
            return {c: 1.0 / n for c in children}
        return {c: raw[c] / total for c in children}

    def _propagate_all_kpis(self):
        """Пересчёт агрегатов для всех KPI снизу вверх + синхронизация представительных значений.
        Использует закешированный порядок обхода (без повторных topological_sort)."""
        if not getattr(self, '_kpi_sub_order', None):
            self._build_propagation_cache()
        for kpi_id in self.kpi_ids:
            self._propagate_for_kpi(kpi_id, self._kpi_sub_order[kpi_id])
        self._sync_agg_value()

    def _propagate_single_kpi(self, kpi_id: str):
        """Свёртка ТОЛЬКО одного KPI (для drop-and-measure: influence/Шепли) — без обхода
        остальных показателей и без синхронизации представительных значений."""
        if not getattr(self, '_kpi_sub_order', None):
            self._build_propagation_cache()
        order = self._kpi_sub_order.get(kpi_id)
        if order is None:
            anc = nx.ancestors(self.G, kpi_id) | {kpi_id}
            order = [n for n in self._topo_order if n in anc]
        self._propagate_for_kpi(kpi_id, order)

    @staticmethod
    def _parse_finances(fin) -> Dict[str, Dict[str, float]]:
        """Надёжный разбор финансового профиля в dict {год: {base, req_extra, add}}.
        Понимает str/двойной JSON/dict; на мусоре возвращает {}."""
        if isinstance(fin, str):
            try:
                fin = json.loads(fin)
                if isinstance(fin, str):
                    fin = json.loads(fin)
            except Exception:
                return {}
        return fin if isinstance(fin, dict) else {}

    def _subtree_leaf_weight_shares(self, parent, desc):
        """Доли листьев поддерева ПО СТРУКТУРНОМУ ВЕСУ связей: сверху вниз распределяем 1.0 от
        parent к листьям по нормированным весам ребёнок→родитель (произведение весов на пути).
        Усредняем по KPI, которые питает parent. Если весов нет — равные доли. Σ долей = 1."""
        if not desc:
            return {}
        sub_nodes = nx.ancestors(self.G, parent) | {parent}
        order = list(reversed([n for n in self._topo_order if n in sub_nodes]))  # родитель раньше детей
        kpis = [k for k in self.kpi_ids if parent in nx.ancestors(self.G, k)]
        acc = {l: 0.0 for l in desc}

        def _run(wmap):
            share = {parent: 1.0}
            for n in order:
                s = share.get(n, 0.0)
                if s <= 0:
                    continue
                preds = [c for c in self.G.predecessors(n) if c in sub_nodes]  # дети (child→parent)
                if not preds:
                    continue
                ws = [max(0.0, float(wmap.get((c, n), {}).get('weight', 0.0))) for c in preds]
                tw = sum(ws)
                if tw <= 1e-9:
                    ws = [1.0] * len(preds); tw = float(len(preds))
                for c, wc in zip(preds, ws):
                    share[c] = share.get(c, 0.0) + s * (wc / tw)
            for l in desc:
                acc[l] += share.get(l, 0.0)

        if kpis:
            for kpi in kpis:
                _run(self.kpi_weights.get(kpi, {}))
            for l in desc:
                acc[l] /= len(kpis)
        tot = sum(acc.values())
        if tot <= 1e-9:
            return {l: 1.0 / len(desc) for l in desc}
        return {l: acc[l] / tot for l in desc}

    def _finances_match(self, a, b, tol: float = 0.5) -> bool:
        """True, если два финансовых профиля совпадают по годам/статусам в пределах допуска
        (матрица показывает эффективные деньги с округлением — точного равенства не будет).
        Используется, чтобы отличить реальную правку денег от повторного применения того же."""
        pa = self._parse_finances(a) or {}
        pb = self._parse_finances(b) or {}
        years = set(pa.keys()) | set(pb.keys())
        for y in years:
            va = pa.get(y, {}) if isinstance(pa.get(y, {}), dict) else {}
            vb = pb.get(y, {}) if isinstance(pb.get(y, {}), dict) else {}
            for st_ in ('base', 'req_extra', 'add'):
                if abs(float(va.get(st_, 0.0) or 0.0) - float(vb.get(st_, 0.0) or 0.0)) > tol:
                    return False
        return True

    def _clear_rollup_sources(self, entity_id: str):
        """Пользователь ЯВНО задал финансы на узле → записи-цели у его ПРЕДКОВ и ПРОМЕЖУТОЧНЫХ
        потомков (не-листьев) устарели и очищаются. Иначе старая цель предка масштабом перетянула
        бы новую правку обратно (например: убрал потребность у листа, а показатель «не изменился»).

        ВАЖНО: перед снятием целей ФИКСИРУЕМ текущее распределение — каждому финансовому листу
        записываем его ЭФФЕКТИВНЫЕ деньги как собственные. Иначе листья, которые не имели своих
        денег и питались целью родителя ПО ВЕСУ, после снятия цели остались бы без источника и
        обнулились (деньги соседей «исчезали» при правке одной работы)."""
        leaves = set(self.get_leaves())
        for L in leaves:
            if L == entity_id:
                continue
            a = self.G.nodes[L]
            eff = self._parse_finances(a.get('finances_eff', {}))
            if not eff:
                continue
            has_money = any(float(v.get(s, 0.0) or 0.0) > 1e-9
                            for v in eff.values() if isinstance(v, dict)
                            for s in ('base', 'req_extra', 'add'))
            if has_money:
                a['finances'] = json.dumps(eff, ensure_ascii=False)   # материализуем то, что видит пользователь
                a['is_financial'] = True
        for anc in nx.descendants(self.G, entity_id):   # предки (рёбра child→parent)
            if str(self.G.nodes[anc].get('type', '')).upper() != 'KPI':
                self.G.nodes[anc]['finances'] = json.dumps({})
        for dsc in nx.ancestors(self.G, entity_id):     # потомки
            if dsc not in leaves and str(self.G.nodes[dsc].get('type', '')).upper() != 'KPI':
                self.G.nodes[dsc]['finances'] = json.dumps({})

    def _compute_effective_finances(self):
        """Пересчитывает ЭФФЕКТИВНЫЕ финансы листьев из ИСТОЧНИКА (введённых значений), не
        разрушая его. Модель: запись на РОДИТЕЛЕ — это ЦЕЛЕВОЙ ИТОГ его поддерева по каждому
        (году, статусу). Листья с собственными деньгами МАСШТАБИРУЮТСЯ к цели пропорционально
        (scale = цель / сумма_листьев); листья без денег получают цель ПО ВЕСУ связей.

        Свойства: ролл-ап (цель == сумме листьев, типично для гос. таблиц с «итого» на каждом
        уровне) → масштаб 1 → нет задвоения (×2/×4); цель = 0 → у листьев 0 (деньги родителя
        убраны); цель выросла/упала → у листьев пропорционально выросло/упало. Источник
        (node['finances']) не меняется; результат в node['finances_eff'] листьев."""
        STATUSES = ('base', 'req_extra', 'add')
        leaves = set(self.get_leaves())
        own = {n: self._parse_finances(self.G.nodes[n].get('finances', {})) for n in self.G.nodes()}
        # старт: эффективные = глубокая копия СОБСТВЕННЫХ финансов листа
        eff = {l: json.loads(json.dumps(own.get(l, {}))) for l in leaves}
        # сверху вниз (родители раньше детей), чтобы вложенные цели каскадились корректно
        order_top_down = [n for n in reversed(self._topo_order) if n not in leaves
                          and str(self.G.nodes[n].get('type', '')).upper() != 'KPI']
        for nid in order_top_down:
            fin = own.get(nid, {})
            if not fin:
                continue
            desc = [n for n in nx.ancestors(self.G, nid) if n in leaves]
            if not desc:
                continue
            # ФИНАНСОВЫЕ СУЩНОСТИ: деньги родителя распределяются ТОЛЬКО на листья, помеченные
            # финансовыми (есть запись в таблице финансов или включён чек-бокс «Финансовая веха»),
            # пропорционально их весу. Нефинансовые вехи денег не получают — иначе перенос сумм
            # между статусами на родителе «поднимал» бы с нуля вехи, у которых денег нет вовсе.
            fin_desc = [l for l in desc if self.G.nodes[l].get('is_financial')]
            if not fin_desc:
                fin_desc = desc  # в поддереве нет ни одной финансовой — фолбэк, чтобы деньги не пропали
            # Весовые доли листьев поддерева; перенормировка на активных в конкретном году
            # финансовых листьях выполняется ниже, внутри годового цикла (wshare_active).
            wshare_all = self._subtree_leaf_weight_shares(nid, desc)

            for y_str, amounts in fin.items():
                if not isinstance(amounts, dict):
                    continue
                    
                try:
                    target_year = int(y_str)
                except ValueError:
                    target_year = None
                    
                # Лист получает деньги года Y, если он НАЧАЛСЯ не позже Y (деньги не могут прийти
                # раньше старта работы). Верхней границы НЕТ: деньги позднего года достаются и
                # работам, которые по плану заканчиваются раньше — их срок затем сдвигает кассовый
                # разрыв на этот год (иначе перенос денег вправо «сжигал» бы деньги у ранних работ).
                active_fin_desc = []
                for l in fin_desc:
                    if target_year is not None:
                        try:
                            s_yr = self._pdate(self.G.nodes[l].get('T_start')).year
                            if s_yr <= target_year:
                                active_fin_desc.append(l)
                        except Exception:
                            active_fin_desc.append(l)  # если даты сломаны, оставляем
                    else:
                        active_fin_desc.append(l)
                
                # Если ничего не пересекается (аномалия сроков), фолбэк на всех, чтобы не сжечь деньги
                if not active_fin_desc:
                    active_fin_desc = fin_desc

                # Пересчитываем весовые доли ТОЛЬКО для активных в этом году
                _ws = {l: wshare_all.get(l, 0.0) for l in active_fin_desc}
                _s = sum(_ws.values())
                wshare_active = ({l: v / _s for l, v in _ws.items()} if _s > 1e-9
                                 else {l: 1.0 / len(active_fin_desc) for l in active_fin_desc})

                for status in STATUSES:
                    if status not in amounts:
                        continue
                    target = float(amounts.get(status, 0.0) or 0.0)
                    own_sum = sum(float(eff.get(l, {}).get(y_str, {}).get(status, 0.0) or 0.0) for l in active_fin_desc)
                    
                    if own_sum > 1e-9:
                        scale = target / own_sum
                        for l in active_fin_desc:
                            if y_str in eff[l] and status in eff[l][y_str]:
                                eff[l][y_str][status] = float(eff[l][y_str][status]) * scale
                    elif target > 1e-9:
                        for l in active_fin_desc:
                            eff[l].setdefault(y_str, {})
                            eff[l][y_str][status] = float(eff[l][y_str].get(status, 0.0) or 0.0) + target * wshare_active.get(l, 0.0)
        for l in leaves:
            self.G.nodes[l]['finances_eff'] = json.dumps(eff[l], ensure_ascii=False)

    def _recompute_parent_budgets(self):
        """F родителя = сумма F листьев-потомков (реальные риск-взвешенные бюджеты). Без этого F
        родителя оставался бы = Σбаза (0, если деньги в «потребности»), что ломает индикатор
        («0 → …») и ЗНАК дельты в сценарии (мнимый рост при снижении финансирования)."""
        leaves = set(self.get_leaves())
        for nid in self.G.nodes():
            if str(self.G.nodes[nid].get('type', '')).upper() == 'KPI' or nid in leaves:
                continue
            desc = [n for n in nx.ancestors(self.G, nid) if n in leaves]
            if desc:
                self.G.nodes[nid]['F'] = float(sum(float(self.G.nodes[l].get('F', 0.0)) for l in desc))
        # срок родителя = по позднему ребёнку (сдвиг кассового разрыва виден и у родителя/в таблице)
        self._propagate_parent_ends()

    def _propagate_parent_ends(self):
        """Родитель завершается, когда завершается его САМЫЙ ПОЗДНИЙ ребёнок: T_end родителя =
        max(его плановый конец, max T_end детей). Так сдвиг срока листа из-за кассового разрыва
        (деньги пришли позже планового года) поднимается вверх по иерархии — и в таблице
        план-график у РОДИТЕЛЯ тоже виден сдвинутый срок, а не только у листа.
        Плановый конец сохраняется в T_plan_end и не теряется."""
        leaves = set(self.get_leaves())
        for nid in self._topo_order:   # дети раньше родителей (топологический порядок child->parent)
            if nid in leaves or str(self.G.nodes[nid].get('type', '')).upper() == 'KPI':
                continue
            kids = [c for c in self.G.predecessors(nid)
                    if str(self.G.nodes[c].get('type', '')).upper() != 'KPI']
            if not kids:
                continue
            self.G.nodes[nid].setdefault('T_plan_end', self.G.nodes[nid].get('T_end'))
            ends = []
            for c in kids:
                ce = self.G.nodes[c].get('T_end')
                if ce:
                    try:
                        ends.append(self._pdate(ce))
                    except Exception:
                        pass
            if not ends:
                continue
            latest = max(ends)
            try:
                own_plan = self._pdate(self.G.nodes[nid].get('T_plan_end')
                                       or self.G.nodes[nid].get('T_end'))
                latest = max(latest, own_plan)
            except Exception:
                pass
            self.G.nodes[nid]['T_end'] = latest.strftime('%Y-%m-%d')

    def _initial_calculate_all(self):
        self._compute_effective_finances()   # эффективные финансы листьев (симметрично, из источника)
        self._recompute_leaf_values()        # активировать финансы (риск/дисконт) + сдвиг сроков
        self._recompute_parent_budgets()     # F родителей = сумма F листьев (+ срок родителя по позднему ребёнку)
        self._propagate_all_kpis()
        self._apply_cashgap_baseline_dip()   # базовый U-провал от кассовых разрывов (Шаг 3)

    # ----- API для экспертного редактора весов -----
    def iter_weight_rows(self, with_influence: bool = True) -> List[Dict[str, Any]]:
        """Плоский список всех весов связей по каждому KPI для отображения/редактирования.

        Для наглядности рядом с ВЕСОМ (настраиваемый вход) показывается ВЛИЯНИЕ источника
        на KPI (измеряемый выход, = share): см. kpi_influence. Влияние мемоизируется по
        (KPI, источник), чтобы не пересчитывать «обнуление» многократно."""
        rows = []
        infl_cache: Dict[Tuple[str, str], float] = {}
        for kpi_id, wd in self.kpi_weights.items():
            kpi_name = str(self.G.nodes[kpi_id].get('name', kpi_id)) if kpi_id in self.G.nodes else kpi_id
            for (s, t), d in wd.items():
                s_name = str(self.G.nodes[s].get('name', s)) if s in self.G.nodes else s
                t_name = str(self.G.nodes[t].get('name', t)) if t in self.G.nodes else t
                rt = canonical_relation(d.get('relation_type'))
                if with_influence and s in self.G.nodes:
                    key = (kpi_id, s)
                    if key not in infl_cache:
                        infl_cache[key] = self.kpi_influence(s, kpi_id)
                    influence = infl_cache[key]
                else:
                    influence = None
                rows.append({
                    'kpi_id': kpi_id, 'KPI': kpi_name,
                    'source_id': s, 'Источник': f"{s} — {s_name}",
                    'target_id': t, 'Родитель': f"{t} — {t_name}",
                    'Вес': round(float(d.get('weight', 0.0)), 4),
                    'Влияние': (round(influence, 4) if influence is not None else None),
                    'relation_type': rt,
                    'Тип': RELATION_LABELS.get(rt, RELATION_LABELS['linear']),
                    'Обоснование': d.get('rationale', ''),
                })
        return rows

    def _graph_kpi_signature(self, kpi_id: str) -> str:
        """Структурная подпись подграфа KPI по текущему графу (для сохранения правок)."""
        ancestors = nx.ancestors(self.G, kpi_id) | {kpi_id}
        edges = sorted((str(u), str(v)) for u, v in self.G.edges()
                       if u in ancestors and v in ancestors)
        node_ids = sorted(ancestors)
        meta = [f"{n}|{self.G.nodes[n].get('type','')}|{self.G.nodes[n].get('name','')}" for n in node_ids]
        meth = self.methodologies.get(kpi_id, "") or ""
        payload = json.dumps({'edges': edges, 'nodes': meta, 'meth': hashlib.md5(meth.encode('utf-8')).hexdigest()},
                             ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    def set_weight(self, kpi_id: str, source: str, target: str, weight: float, relation_type: str = None, rationale: str = None):
        prev = self.kpi_weights.get(kpi_id, {}).get((str(source), str(target)), {})
        rt = canonical_relation(relation_type) if relation_type is not None else prev.get('relation_type', 'manual')
            
        # Если передан новый комментарий, используем его. Иначе берем старый (или дефолтный)
        rat = rationale if rationale is not None else prev.get('rationale', 'ручная правка эксперта')
            
        self.kpi_weights.setdefault(kpi_id, {})[(str(source), str(target))] = {
            'weight': float(weight),
            'relation_type': rt,
            'rationale': rat,
        }
            # ДОБАВЛЕНО (#11): правка эксперта сразу пишется в пофайловый кеш весов KPI.
        self._save_kpi_weights_file(kpi_id, self._graph_kpi_signature(kpi_id))

    def recalculate(self):
        """Полный пересчёт агрегатов и проверок (после ручных правок весов/параметров/финансов)."""
        self._compute_quarter_windows()
        self._compute_effective_finances()   # пересобрать эффективные финансы листьев из источника
        self._recompute_leaf_values()        # значения листьев с учётом финансов (симметрично)
        self._recompute_parent_budgets()     # F родителей = сумма F листьев
        self._propagate_all_kpis()
        self._apply_cashgap_baseline_dip()   # обновить базовый U-провал (кассовые разрывы)
        self._validate_budgets()
        self._validate_schedule()

    # ----- Снапшоты состояния (для долговечного базового плана проекта) -----
    def _recompute_leaf_values(self):
        """Пересчитать локальную ценность всех листьев из их F/сроков. Деньги — ТОЛЬКО из
        финансового профиля (отдельная таблица); бюджет из план-графика нигде не используется —
        нет записи в финансах → F=0 (это ощутимо, а не «тихо подставляем старое число»)."""
        for n in self.get_leaves():
            a = self.G.nodes[n]
            try:
                a.setdefault('T_plan_end', a.get('T_end')) 
                # ЭФФЕКТИВНЫЕ финансы листа (собственные + распределённые от родителей); если их
                # ещё не считали — собственные. Источник node['finances'] при этом не меняется.
                fin = a.get('finances_eff', a.get('finances', {}))
                r_req = float(a.get('rho_req', 1.0))
                r_add = float(a.get('rho_add', 0.0))
                f_eff, f_real, last_year = self._evaluate_node_finances(fin, r_req, r_add)
                has_fin = (f_eff > 1e-9) or (f_real > 1e-9) or (last_year is not None)

                if not has_fin:
                    a['F'] = 0.0
                    F_use = 0.0
                    # нет денег — срок возвращается к плановому (снимаем прежний сдвиг разрыва)
                    actual_end = a.get('T_plan_end') or a.get('T_end')
                    a['T_end'] = actual_end
                else:
                    a['F'] = f_eff  
                    F_use = f_real  
                    # Сдвиг считаем ОТ ПЛАНОВОГО срока (T_plan_end), а не от текущего (возможно уже
                    # сдвинутого) T_end — иначе сдвиг был бы односторонним и не откатывался при
                    # возврате денег в ранний год. T_end = max(плановый, 31.12 последнего года с деньгами).
                    plan_end_s = a.get('T_plan_end') or a.get('T_end')
                    plan_end = self._pdate(plan_end_s)
                    actual_end = plan_end_s
                    if last_year and last_year > plan_end.year:
                        try:
                            actual_end = datetime(last_year, 12, 31).strftime('%Y-%m-%d')
                        except ValueError:
                            actual_end = plan_end_s
                    a['T_end'] = actual_end

                if self._is_milestone_type(a.get('type')):
                    a['local_value'] = self._milestone_value(n, F_use, actual_end)
                else:
                    d = max(1, (self._pdate(actual_end) - self._pdate(a['T_start'])).days)
                    a['local_value'] = self._calculate_local_value(F_use, d, T_opt=a.get('T_opt', d), late_days=self._late_days(a, actual_end))
            except Exception:
                pass

    def _apply_cashgap_baseline_dip(self):
        """БАЗОВЫЙ U-провал от кассовых разрывов (Шаг 3, автоматически, без сценария).

        Если работа профинансирована ПОЗЖЕ планового срока (её T_end сдвинут за T_plan_end),
        показатель проседает в кварталах ОЖИДАНИЯ (плановый квартал ≤ квартал < фактический)
        на величину влияния работы и восстанавливается с фактического срока. Меняет только
        БАЗОВЫЕ periods/annual показателя (то, что видно на загрузке до запуска сценария).
        Прошлые кварталы тоже проседают — недостигнутая веха не становится достигнутой оттого,
        что квартал закрылся. Для проектов без разрывов — no-op."""
        for kpi_id in self.kpi_ids:
            periods = self._kpi_periods(kpi_id)
            if not periods:
                continue
            # листья-работы этого KPI с РЕАЛЬНЫМ сдвигом срока (кассовый разрыв)
            sub = nx.ancestors(self.G, kpi_id)
            leaves = [n for n in sub
                      if str(self.G.nodes[n].get('type', '')).upper() != 'KPI' and self.G.in_degree(n) == 0]
            shifted = []
            for l in leaves:
                a = self.G.nodes[l]
                pe = a.get('T_plan_end') or a.get('T_end')
                ae = a.get('T_end')
                try:
                    pq = self._quarter_of(pe); aq = self._quarter_of(ae)
                except Exception:
                    continue
                if pq[0] is None or aq[0] is None or aq <= pq:
                    continue  # не сдвинута — провала нет
                infl = self.kpi_influence(l, kpi_id)
                if infl > 1e-9:
                    shifted.append((pq, aq, infl))
            # пересчитываем базовый прогноз ОТ ПЛАНА (идемпотентно).
            # Прошлые кварталы тоже проседают: если веху планировали к Q1, а деньги пришли позже,
            # то результата в Q1 фактически не было — показывать там план значило бы приукрашивать.
            for p in periods:
                yq = (int(p['year']), int(p['q']))
                dip = min(1.0, sum(infl for (pq, aq, infl) in shifted if pq <= yq < aq))
                nf = self._clamp_kpi_forecast(kpi_id, float(p['plan']) * (1.0 - dip))
                p['forecast'] = round(nf, 4)
                p['deviation'] = round(nf - float(p['plan']), 4)
                p['changed'] = dip > 1e-9
            self.G.nodes[kpi_id]['periods'] = json.dumps(periods, ensure_ascii=False)
            # годовой блок: провал по состоянию на конец года
            annual = self._kpi_annual(kpi_id)
            if annual:
                for yr, av in annual.items():
                    yq = (int(yr), 4)
                    dip = min(1.0, sum(infl for (pq, aq, infl) in shifted if pq <= yq < aq))
                    av['forecast'] = round(self._clamp_kpi_forecast(kpi_id, float(av.get('plan', 0.0)) * (1.0 - dip)), 4)
                self.G.nodes[kpi_id]['annual'] = json.dumps({str(y): v for y, v in annual.items()}, ensure_ascii=False)

    def export_state(self) -> Dict[str, Any]:
        """Сериализуемое состояние утверждённого плана: бюджеты/сроки задач и периоды KPI.

        Веса связей хранятся отдельно (в weights_matrix.json проекта), поэтому в снапшот
        не входят — он описывает именно утверждённый план."""
        tasks, kpis = {}, {}
        for n, a in self.G.nodes(data=True):
            if str(a.get('type', '')).upper() == 'KPI':
                kpis[n] = {
                    'periods': a.get('periods'),
                    'annual': a.get('annual'),
                    'Q1': a.get('Q1'), 'Q2': a.get('Q2'), 'Q3': a.get('Q3'),
                    'Q4': a.get('Q4'), 'Year': a.get('Year'),
                }
            else:
                fin = a.get('finances')
                if isinstance(fin, str):
                    try:
                        fin = json.loads(fin)
                    except Exception:
                        fin = {}
                tasks[n] = {'F': a.get('F'), 'T_start': a.get('T_start'),
                            'T_end': a.get('T_end'), 'T_opt': a.get('T_opt'),
                            'T_plan_end': a.get('T_plan_end'),
                            # ИСТОЧНИК денег узла — обязателен в снапшоте: без него утверждённый
                            # перенос/ввод средств (base↔потребность, между годами) не переживал
                            # пересборку движка и откатывался к значениям из Excel. Риск-веса
                            # rho_req/rho_add и флаг финансовой вехи тоже относятся к плану.
                            'finances': fin,
                            'is_financial': bool(a.get('is_financial', False)),
                            'rho_req': a.get('rho_req', 1.0),
                            'rho_add': a.get('rho_add', 0.0)}
        return {'version': 1, 'tasks': tasks, 'kpis': kpis}

    def apply_state(self, state: Optional[Dict[str, Any]]):
        """Наложить снапшот поверх собранного из Excel графа (структура та же).

        Применяется после инициализации движка: восстанавливает утверждённые бюджеты,
        сроки и квартальные значения, затем полностью пересчитывает модель."""
        if not state:
            return
        for n, vals in (state.get('tasks') or {}).items():
            if n in self.G.nodes and str(self.G.nodes[n].get('type', '')).upper() != 'KPI':
                node = self.G.nodes[n]
                for k in ('F', 'T_start', 'T_end', 'T_opt', 'T_plan_end', 'rho_req', 'rho_add'):
                    if vals.get(k) is not None:
                        node[k] = vals[k]
                # Источник денег (если сохранён — новые снапшоты). Старые снапшоты его не
                # содержат: тогда оставляем финансы из Excel (обратная совместимость).
                if vals.get('finances') is not None:
                    node['finances'] = vals['finances']
                if 'is_financial' in vals:
                    node['is_financial'] = bool(vals['is_financial'])
        for n, vals in (state.get('kpis') or {}).items():
            if n in self.G.nodes and str(self.G.nodes[n].get('type', '')).upper() == 'KPI':
                node = self.G.nodes[n]
                for k in ('periods', 'annual', 'Q1', 'Q2', 'Q3', 'Q4', 'Year'):
                    if vals.get(k) is not None:
                        node[k] = vals[k]
        # Пересобрать деньги ИЗ ВОССТАНОВЛЕННОГО ИСТОЧНИКА: эффективные финансы листьев,
        # их ценность и бюджеты родителей — тем же конвейером, что и recalculate(). Без
        # _compute_effective_finances() лист брал бы finances_eff, посчитанные при сборке из
        # Excel, и утверждённый перенос средств терялся. Кассовый U-провал (_apply_cashgap…)
        # НЕ вызываем — квартальные периоды KPI берём как утверждено (из снапшота).
        self._compute_quarter_windows()
        self._compute_effective_finances()
        self._recompute_leaf_values()
        self._recompute_parent_budgets()
        self._propagate_all_kpis()
        self._validate_budgets()
        self._validate_schedule()


    def _quarter_coverage(self, s, e) -> List[float]:
        """Доля каждой фазы проекта, покрытая окном [s, e] работы (значения в [0, 1])."""
        if not self.quarter_windows:
            return []
        try:
            s, e = self._pdate(s), self._pdate(e)
        except Exception:
            return [0.0] * len(self.quarter_windows)
        if e < s:
            s, e = e, s
        cov = []
        for (a, b) in self.quarter_windows:
            L = max(1, (b - a).days)
            overlap = (min(e, b) - max(s, a)).days
            cov.append(min(1.0, max(0.0, overlap) / L))
        return cov

    # ----- Реальные календарные кварталы и трёхдатная проекция -----
    @staticmethod
    def _quarter_dates(year: int, q: int):
        """Границы реального календарного квартала."""
        starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        return datetime(year, *starts[q]), datetime(year, *ends[q])

    def _kpi_periods(self, kpi_id: str) -> List[Dict[str, Any]]:
        """Список квартальных периодов KPI (мультигод): [{year, q, plan, forecast, fact}]."""
        raw = self.G.nodes[kpi_id].get('periods')
        if isinstance(raw, str):
            try: return json.loads(raw)
            except Exception: return []
        return raw if isinstance(raw, list) else []

    def _kpi_annual(self, kpi_id: str) -> Dict[int, Dict[str, float]]:
        """Годовой план/прогноз/факт KPI по годам (#1): {year: {plan, forecast, fact}}.

        Берётся из отдельного годового блока таблицы, НЕ как сумма кварталов."""
        raw = self.G.nodes[kpi_id].get('annual')
        out = {}
        if isinstance(raw, str):
            try:
                out = {int(yr): v for yr, v in json.loads(raw).items()}
            except Exception:
                out = {}
        elif isinstance(raw, dict):
            out = {int(yr): v for yr, v in raw.items()}
        return out

    @staticmethod
    def _window_overlap_fraction(ws, we, ps, pe) -> float:
        """Доля периода [ps, pe], покрытая окном [ws, we]."""
        L = max(1, (pe - ps).days)
        ov = (min(we, pe) - max(ws, ps)).days
        return min(1.0, max(0.0, ov) / L)

    def kpi_influence(self, node: str, kpi_id: str) -> float:
        """ЕДИНОЕ понятие ВЛИЯНИЯ: насколько (в долях [0,1]) просядет значение KPI, если
        убрать вклад узла (обнулить ценность листьев его поддерева) — неразрушающе.

        Это один измеряемый примитив для всей модели:
        • на уровне KPI это «доля участия» работы (то, что раньше называлось share);
        • рычаг и эластичность — то же влияние, выраженное на единицу/в процентах бюджета.
        Вес связи (weight) — это НАСТРАИВАЕМЫЙ ВХОД, который вместе с нелинейной агрегацией
        ФОРМИРУЕТ это влияние; влияние — измеряемый ВЫХОД."""
        base = self._kpi_value(kpi_id)
        if base <= 1e-9:
            return 0.0
        sub = nx.ancestors(self.G, node) | {node}
        leaves = [n for n in sub
                  if str(self.G.nodes[n].get('type', '')).upper() != 'KPI' and self.G.in_degree(n) == 0]
        if not leaves:
            return 0.0
        snap = {l: self.G.nodes[l].get('local_value', 0.0) for l in leaves}
        try:
            for l in leaves:
                self.G.nodes[l]['local_value'] = 0.0
            self._propagate_single_kpi(kpi_id)   # нужен только этот KPI — не все
            without = self._kpi_value(kpi_id)
        finally:
            for l, lv in snap.items():
                self.G.nodes[l]['local_value'] = lv
            self._propagate_single_kpi(kpi_id)   # восстановить только затронутый KPI
        return float(min(1.0, max(0.0, (base - without) / base)))

    def _entity_kpi_share(self, entity_id: str, kpi_id: str) -> float:
        """share === влияние на уровне KPI (см. kpi_influence). Имя сохранено для совместимости."""
        return self.kpi_influence(entity_id, kpi_id)

    def _quarter_of(self, d) -> tuple:
        """Возвращает (год, квартал) для даты — квартал, в котором работа ЗАВЕРШАЕТСЯ."""
        try:
            dt = self._pdate(d)
        except Exception:
            return (None, None)
        return (dt.year, (dt.month - 1) // 3 + 1)

    def project_periods(self, win_old, win_new, kpi_id: str, m: float, share: float, today=None, children_ends=None):
        """Поквартальная проекция KPI с учётом СТАРОГО и НОВОГО квартала завершения работы.

        Три зоны (для открытых кварталов), где q_old/q_new — кварталы завершения по плану и по сценарию:
          • до ожидаемого срока (yq < q_old и yq < q_new): результат ещё не ждали → прогноз = план;
          • «провал ожидания» (q_old ≤ yq < q_new): результат ПЛАНИРОВАЛИ к этому кварталу, но он
            задержан → показатель проседает на величину влияния работы: план × (1 − влияние);
          • с фактического (нового) срока (yq ≥ q_new): результат достигнут с новыми параметрами →
            план × m.
        При урезании бюджета без сдвига срока q_old == q_new и средняя зона исчезает (как раньше).
        ПРОШЛЫЕ кварталы не замораживаются: если работа уже завершилась, изменение её денег/сроков
        меняет и показатель прошлых кварталов (ретроспективный «что если»). Факт (поле fact) при
        этом не трогается — он остаётся мерилом того, что произошло на самом деле. Кварталы, в
        которых результата ещё не ждали, и так получают множитель 1 по зональному правилу.
        Год считается по тому же правилу — по состоянию на конец года (последний квартал года).

        children_ends — список дат окончания дочерних задач; если есть дети, завершающиеся раньше
        родителя, эффект начинается с самой ранней даты (дети начинают снижать KPI раньше
        родителя)."""
        periods = self._kpi_periods(kpi_id)
        if not periods:
            return [], {}
        today = today or datetime.now()
        try:
            q_end_old = self._quarter_of(win_old[1])  # квартал завершения ПО ПЛАНУ
            q_end_new = self._quarter_of(win_new[1])  # квартал завершения ПО СЦЕНАРИЮ
        except Exception:
            q_end_old = q_end_new = (None, None)
        # Дети, завершающиеся раньше родителя — их эффект начинается раньше
        if children_ends:
            try:
                _qs = [self._quarter_of(dd) for dd in children_ends if dd]
                _qs = [q for q in _qs if q[0] is not None]   # отбросить неразобранные даты
                if _qs:
                    _earliest = min(_qs)
                    if q_end_new[0] is None or _earliest < q_end_new:
                        q_end_new = _earliest
            except Exception:
                pass
        share = max(0.0, min(1.0, float(share)))

        def zone_mult(yq):
            """Множитель к плану для квартала yq=(год,квартал) по трём зонам."""
            delivered_new = (q_end_new[0] is None) or (yq >= q_end_new)
            if delivered_new:
                return m                      # результат достигнут (новые параметры)
            expected_old = (q_end_old[0] is not None) and (yq >= q_end_old)
            if expected_old:
                return max(0.0, 1.0 - share)  # ждали результат, а он задержан → провал на влияние
            return 1.0                        # ещё не ждали → по плану

        dates = [self._quarter_dates(int(p['year']), int(p['q'])) for p in periods]
        plan = [float(p.get('plan', 0.0)) for p in periods]
        future = [pe >= today for (ps, pe) in dates]

        out = []
        for i, p in enumerate(periods):
            fact = parse_ru_number(p.get('fact', 0.0), 0.0)
            yq = (int(p['year']), int(p['q']))

            # Прошлое НЕ замораживаем: если работа уже завершилась (её квартал в прошлом), то
            # изменение её денег/сроков меняет и показатель тех кварталов — это ретроспективный
            # «что если», а не подделка факта (факт хранится отдельно, в поле fact).
            # Квартал, до которого результата ещё не ждали, и так получит множитель 1 по зонам.
            mq = zone_mult(yq)
            forecast = max(0.0, plan[i] * mq)
            changed = abs(mq - 1.0) > 1e-9

            out.append({
                'year': int(p['year']), 'q': int(p['q']),
                'label': f"{int(p['q'])} кв. {int(p['year'])}",
                'plan': round(plan[i], 4), 'forecast': round(forecast, 4), 'fact': round(fact, 4),
                'deviation': round(forecast - plan[i], 4),
                'locked': False, 'changed': changed,
                'past': not future[i],      # квартал уже прошёл — интерфейс покажет как ретроспективу
            })
        # Годовое значение — из ОТДЕЛЬНОГО годового плана (не сумма кварталов, #1);
        # прогноз года = годовой план × m (изменение ценности KPI).
        annual_plan = self._kpi_annual(kpi_id)
        years = sorted({int(p['year']) for p in periods})
        annual = {}
        for yr in years:
            ap = parse_ru_number(annual_plan.get(yr, {}).get('plan', 0.0), 0.0)
            # Год — по состоянию на конец года (тот же зональный множитель, что и у кварталов):
            # если результат задержан за пределы года, год проседает на влияние, а не на m.
            my = zone_mult((yr, 4))
            annual[yr] = {'plan': round(ap, 4), 'forecast': round(ap * my, 4)}
        return out, annual

    # ═══════════════════ ОБРАТНАЯ ЗАДАЧА: цель по показателю → деньги ═══════════════════
    def _end_quarter(self, node_id: str) -> tuple:
        return self._quarter_of(self.G.nodes[node_id].get('T_end'))

    def target_candidates(self, kpi_id: str, year: int, quarter: int) -> List[Dict[str, Any]]:
        """Работы, которые МОГУТ повлиять на показатель в целевом квартале деньгами:
        финансовые листья, влияющие на KPI и завершающиеся НЕ ПОЗЖЕ целевого квартала.
        Деньги не умеют «ускорять» работу — поэтому те, кто финиширует позже, в список не входят
        (их можно только сдвинуть по срокам вручную)."""
        tq = (int(year), int(quarter))
        out, late = [], []
        for L in self.get_leaves():
            a = self.G.nodes[L]
            if str(a.get('type', '')).upper() == 'KPI':
                continue
            infl = self.kpi_influence(L, kpi_id)
            if infl <= 1e-9:
                continue
            eq = self._end_quarter(L)
            row = {'id': L, 'name': str(a.get('name', L)), 'influence': round(infl, 4),
                   'end': a.get('T_end'), 'F': float(a.get('F', 0.0)),
                   'is_financial': bool(a.get('is_financial', False))}
            if eq[0] is not None and eq <= tq:
                if row['is_financial']:
                    out.append(row)
            else:
                late.append(row)
        out.sort(key=lambda r: -r['influence'])
        late.sort(key=lambda r: -r['influence'])
        return out, late

    def solve_for_target(self, kpi_id: str, year: int, quarter: int, target_value: float,
                         mode: str = 'proportional', entity_id: str = None,
                         max_scale: float = 25.0, tol: float = 1e-4, max_iter: int = 60) -> Dict[str, Any]:
        """ОБРАТНЫЙ СЦЕНАРИЙ. Дано: показатель, год, квартал и ЖЕЛАЕМОЕ значение.
        Найти: сколько денег нужно добавить (или сколько высвободится) и как сдвинется план-график.

        Математика (обращение прямой модели):
          прогноз = план × m_калибр.          → m_нужное = цель / план
          m_калибр. = 1 + conf·s·(m_сырое−1)  → m_сырое  = 1 + (m_нужное − 1)/(conf·s)
          m_сырое = V_нов / V_стар            → V_цель   = m_сырое · V_текущее
        Ценность монотонно растёт с деньгами (логарифм) ⇒ нужный бюджет ищется бинарным поиском.

        Неразрушающе: состояние снимается и восстанавливается; возвращается ПЛАН изменений."""
        res = {'feasible': False, 'reason': '', 'kpi': kpi_id, 'year': int(year), 'quarter': int(quarter)}

        periods = self._kpi_periods(kpi_id)
        row = next((p for p in periods if int(p['year']) == int(year) and int(p['q']) == int(quarter)), None)
        if not row:
            res['reason'] = 'В плане показателя нет такого квартала.'
            return res
        plan = float(row.get('plan', 0.0) or 0.0)
        if plan <= 1e-9:
            res['reason'] = 'Плановое значение квартала равно нулю — от него нельзя посчитать множитель.'
            return res

        # Прошлый квартал допустим: это РЕТРОСПЕКТИВНЫЙ вопрос («сколько денег не хватило, чтобы
        # показатель в том квартале вышел на цель»). Помечаем результат, чтобы интерфейс пояснил.
        retrospective = False
        try:
            _, q_end_date = self._quarter_dates(int(year), int(quarter))
            retrospective = q_end_date < datetime.now()
        except Exception:
            pass
        res['retrospective'] = retrospective

        target = float(target_value)
        m_need = target / plan

        cal = self.get_kpi_calibration(kpi_id)
        denom = float(cal.get('confidence', 1.0)) * float(cal.get('sensitivity', 1.0))
        if denom <= 1e-6:
            res['reason'] = 'Доверие/чувствительность показателя равны нулю — модель не реагирует на изменения.'
            return res
        m_raw = 1.0 + (m_need - 1.0) / denom

        V_old = self._kpi_value(kpi_id)
        if V_old <= 1e-9:
            res['reason'] = 'Текущая ценность показателя равна нулю (нет весов/денег) — обратный расчёт невозможен.'
            return res
        V_target = m_raw * V_old
        if V_target <= 0:
            res['reason'] = 'Целевое значение требует отрицательной ценности — недостижимо.'
            return res

        cands, late = self.target_candidates(kpi_id, year, quarter)
        if entity_id:
            cands = [c for c in cands if c['id'] == entity_id]
            if not cands:
                res['reason'] = 'Выбранная работа не финансовая, не влияет на показатель или завершается позже квартала.'
                res['late'] = late
                return res
        if not cands:
            res['reason'] = ('Ни одна финансовая работа, влияющая на показатель, не завершается к этому кварталу. '
                             'Деньги не могут ускорить результат — можно только сдвинуть сроки работ.')
            res['late'] = late
            return res
        if mode == 'single' and not entity_id:
            cands = cands[:1]

        ids = [c['id'] for c in cands]
        snapshot = {n: {'F': self.G.nodes[n].get('F', 0.0),
                        'T_start': self.G.nodes[n].get('T_start', ''),
                        'T_end': self.G.nodes[n].get('T_end', ''),
                        'local_value': self.G.nodes[n].get('local_value', 0.0),
                        'finances': copy.deepcopy(self.G.nodes[n].get('finances', {})),
                        'finances_eff': copy.deepcopy(self.G.nodes[n].get('finances_eff', None)),
                        'is_financial': self.G.nodes[n].get('is_financial', False)}
                    for n in self.G.nodes()
                    if str(self.G.nodes[n].get('type', '')).upper() != 'KPI'}

        def _scale(fin: dict, k: float) -> dict:
            out = {}
            for y, v in (fin or {}).items():
                if isinstance(v, dict):
                    out[str(y)] = {s: float(v.get(s, 0.0) or 0.0) * k for s in ('base', 'req_extra', 'add')}
            return out

        try:
            # ФИКСИРУЕМ текущее распределение (материализация) и снимаем цели предков —
            # дальше деньги кандидатов можно менять, не задевая остальные работы.
            self._clear_rollup_sources(ids[0])
            self._compute_effective_finances()
            base = {L: self._parse_finances(self.G.nodes[L].get('finances_eff',
                                                                self.G.nodes[L].get('finances', {})))
                    for L in ids}
            ends_before = {n: self.G.nodes[n].get('T_end') for n in snapshot}
            money_before = sum(float(self.G.nodes[L].get('F', 0.0)) for L in ids)

            def V_at(k: float) -> float:
                for L in ids:
                    self.G.nodes[L]['finances'] = json.dumps(_scale(base[L], k), ensure_ascii=False)
                    self.G.nodes[L]['is_financial'] = True
                self._compute_effective_finances()
                self._recompute_leaf_values()
                self._recompute_parent_budgets()
                self._propagate_all_kpis()
                return self._kpi_value(kpi_id)

            v_lo, v_hi = V_at(0.0), V_at(max_scale)
            if V_target > v_hi + 1e-9:
                res.update({'reason': 'Цель недостижима одними деньгами: показатель насыщается (логарифм).',
                            'best_value': round(v_hi, 4),
                            'best_forecast': round(plan * (1.0 + denom * (v_hi / V_old - 1.0)), 4),
                            'late': late})
                return res
            if V_target < v_lo - 1e-9:
                res.update({'reason': 'Цель ниже минимума: даже при нулевом бюджете этих работ показатель выше — '
                                      'вклад дают другие работы.',
                            'floor_value': round(v_lo, 4),
                            'floor_forecast': round(plan * (1.0 + denom * (v_lo / V_old - 1.0)), 4),
                            'late': late})
                return res

            lo, hi = 0.0, max_scale
            for _ in range(max_iter):
                mid = (lo + hi) / 2.0
                if V_at(mid) < V_target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            k_star = (lo + hi) / 2.0
            V_new = V_at(k_star)

            money_after = sum(float(self.G.nodes[L].get('F', 0.0)) for L in ids)
            per_work, by_year = [], {}
            for L in ids:
                f_b = self._parse_finances(base[L])
                f_a = _scale(f_b, k_star)
                tot_b = sum(sum(float(v.get(s, 0.0) or 0.0) for s in ('base', 'req_extra', 'add'))
                            for v in f_b.values())
                tot_a = sum(sum(float(v.get(s, 0.0) or 0.0) for s in ('base', 'req_extra', 'add'))
                            for v in f_a.values())
                per_work.append({'id': L, 'name': str(self.G.nodes[L].get('name', L)),
                                 'influence': round(self.kpi_influence(L, kpi_id), 4),
                                 'before': round(tot_b, 2), 'after': round(tot_a, 2),
                                 'delta': round(tot_a - tot_b, 2),
                                 'end': self.G.nodes[L].get('T_end')})
                for y, v in f_a.items():
                    d_y = sum(float(v.get(s, 0.0) or 0.0) for s in ('base', 'req_extra', 'add'))
                    b_y = sum(float(f_b.get(y, {}).get(s, 0.0) or 0.0) for s in ('base', 'req_extra', 'add'))
                    acc = by_year.setdefault(str(y), {'before': 0.0, 'after': 0.0})
                    acc['before'] += b_y
                    acc['after'] += d_y
            for y in by_year:
                by_year[y]['delta'] = round(by_year[y]['after'] - by_year[y]['before'], 2)
                by_year[y]['before'] = round(by_year[y]['before'], 2)
                by_year[y]['after'] = round(by_year[y]['after'], 2)

            sched = []
            for n, e0 in ends_before.items():
                e1 = self.G.nodes[n].get('T_end')
                if e0 != e1:
                    sched.append({'id': n, 'name': str(self.G.nodes[n].get('name', n)),
                                  'end_before': e0, 'end_after': e1})

            side = [{'kpi': k2, 'value_after': round(self._kpi_value(k2), 4)}
                    for k2 in self.kpi_ids if k2 != kpi_id]

            m_raw_ach = V_new / V_old if V_old > 0 else 1.0
            m_cal_ach = self._calibrate_m(kpi_id, m_raw_ach)

            res.update({
                'feasible': True,
                'direction': 'add' if (money_after - money_before) > 1e-6 else
                             ('free' if (money_after - money_before) < -1e-6 else 'none'),
                'plan': round(plan, 4), 'target': round(target, 4),
                'forecast_now': round(plan * self._calibrate_m(kpi_id, 1.0), 4),
                'forecast_after': round(plan * m_cal_ach, 4),
                'm_needed': round(m_need, 5), 'm_achieved': round(m_cal_ach, 5),
                'V_old': round(V_old, 4), 'V_target': round(V_target, 4), 'V_new': round(V_new, 4),
                'scale': round(k_star, 5),
                'money_before': round(money_before, 2), 'money_after': round(money_after, 2),
                'money_delta': round(money_after - money_before, 2),
                'per_work': sorted(per_work, key=lambda r: -abs(r['delta'])),
                'by_year': by_year, 'schedule': sched, 'late': late, 'mode': mode,
                'side_effects': side,
            })
            return res
        finally:
            for n, s in snapshot.items():
                self.G.nodes[n].update(s)
            self._compute_effective_finances()
            self._recompute_leaf_values()
            self._recompute_parent_budgets()
            self._propagate_all_kpis()

    def apply_target_solution(self, sol: Dict[str, Any]) -> bool:
        """Применяет найденный обратным расчётом план: записывает деньги работам и пересчитывает."""
        if not sol or not sol.get('feasible') or not sol.get('per_work'):
            return False
        k = float(sol.get('scale', 1.0))
        ids = [w['id'] for w in sol['per_work']]
        if not ids:
            return False
        self._clear_rollup_sources(ids[0])
        self._compute_effective_finances()
        for L in ids:
            fin = self._parse_finances(self.G.nodes[L].get('finances_eff',
                                                           self.G.nodes[L].get('finances', {})))
            scaled = {str(y): {s: float(v.get(s, 0.0) or 0.0) * k for s in ('base', 'req_extra', 'add')}
                      for y, v in fin.items() if isinstance(v, dict)}
            self.G.nodes[L]['finances'] = json.dumps(scaled, ensure_ascii=False)
            self.G.nodes[L]['is_financial'] = True
        self._compute_effective_finances()
        self._recompute_leaf_values()
        self._recompute_parent_budgets()
        self._propagate_all_kpis()
        self._apply_cashgap_baseline_dip()
        return True

    # ----- Аналитика: чувствительность и оптимизация бюджета -----
    def get_leaves(self) -> List[str]:
        """Листовые работы (Вехи и т.п.) — те, у кого нет своих подзадач (in_degree == 0)."""
        return [n for n, a in self.G.nodes(data=True)
                if str(a.get('type', '')).upper() != 'KPI' and self.G.in_degree(n) == 0]

    def sensitivity_analysis(self, budget_bump_pct: float = 0.05, days_bump: int = 30,
                             min_bump: float = 1.0, eps: float = 1e-9) -> List[Dict[str, Any]]:
        """Предельная отдача каждой листовой работы по каждому KPI, который она питает.

        Через неразрушающий mix() оцениваются конечные разности:
        - dKPI_dF       — прирост KPI на единицу бюджета (рычаг);
        - elasticity_F  — эластичность (%ΔKPI на %ΔF), безразмерна и сравнима между работами;
        - dKPI_per_month — изменение KPI при продлении срока на days_bump дней.
        Результат отсортирован по убыванию рычага."""
        base = {k: self._kpi_value(k) for k in self.kpi_ids}
        rows = []
        for leaf in self.get_leaves():
            node = self.G.nodes[leaf]
            F0 = float(node.get('F', 0.0))
            s0, e0 = node.get('T_start'), node.get('T_end')
            dF = max(min_bump, F0 * budget_bump_pct)
            fin_leaf = self._is_financial_leaf(leaf)   # деньги можно давать только финансовым работам
            resF = None
            if fin_leaf:
                try:
                    resF = self.mix(leaf, F0 + dF, s0, e0, project=False)
                except Exception as ex:
                    logger.warning(f"Чувствительность: пропуск {leaf} (бюджет): {ex}")
                    resF = None
            resT = None
            try:
                e1 = (self._pdate(e0) + timedelta(days=days_bump)).strftime("%Y-%m-%d")
                resT = self.mix(leaf, F0, s0, e1, project=False)
            except Exception:
                resT = None
            fed = set(self.get_node_kpis(leaf))
            for kpi in self.kpi_ids:
                if kpi not in fed:
                    continue
                b = base[kpi]
                dV_F = (resF[kpi]['new'] - b) if resF else 0.0
                dV_T = (resT[kpi]['new'] - b) if resT else 0.0
                rows.append({
                    'leaf': leaf, 'leaf_name': node.get('name', leaf),
                    'kpi_id': kpi, 'kpi_name': self.G.nodes[kpi].get('name', kpi),
                    'F': F0, 'is_financial': fin_leaf,
                    'dKPI_dF': dV_F / dF if dF > eps else 0.0,
                    'elasticity_F': ((dV_F / b) / (dF / F0)) if (b > eps and F0 > eps) else 0.0,
                    'dKPI_per_month': dV_T,
                    'leverage': dV_F / dF if dF > eps else 0.0,
                })
        rows.sort(key=lambda r: -r['leverage'])
        return rows

    def _simulate_assignment(self, assignments: Dict[str, float]) -> Dict[str, float]:
        """Неразрушающе применяет новые бюджеты к набору листьев и возвращает значения KPI.

        Деньги пишутся В ФИНАНСЫ (F — производная от них): прямое присвоение node['F'] стиралось
        бы пересчётом, а ценность считалась бы по номиналу вместо дисконтированной суммы."""
        snap = {n: {'F': self.G.nodes[n].get('F', 0.0),
                    'T_end': self.G.nodes[n].get('T_end', ''),
                    'local_value': self.G.nodes[n].get('local_value', 0.0),
                    'finances': copy.deepcopy(self.G.nodes[n].get('finances', {})),
                    'finances_eff': copy.deepcopy(self.G.nodes[n].get('finances_eff', None)),
                    'is_financial': self.G.nodes[n].get('is_financial', False)}
                for n in self.G.nodes()
                if str(self.G.nodes[n].get('type', '')).upper() != 'KPI'}
        try:
            if assignments:
                self._clear_rollup_sources(next(iter(assignments)))
                self._compute_effective_finances()
            for lid, newF in assignments.items():
                self.G.nodes[lid]['finances'] = json.dumps(self._finances_for_nominal(lid, float(newF)),
                                                           ensure_ascii=False)
                self.G.nodes[lid]['is_financial'] = True
            self._compute_effective_finances()
            self._recompute_leaf_values()
            self._recompute_parent_budgets()
            self._propagate_all_kpis()
            return {k: self._kpi_value(k) for k in self.kpi_ids}
        finally:
            for n, stt in snap.items():
                self.G.nodes[n].update(stt)
            self._compute_effective_finances()
            self._recompute_leaf_values()
            self._recompute_parent_budgets()
            self._propagate_all_kpis()

    def suggest_reallocation(self, kpi_id: str, pool: float, steps: int = 12, top_k: int = 8) -> Dict[str, Any]:
        """Жадное распределение дополнительного бюджета pool ради максимума kpi_id.

        На каждом шаге добавляет порцию бюджета той работе, что даёт наибольший
        прирост KPI ЗДЕСЬ И СЕЙЧАС. Так как отдача от бюджета убывающая (ln),
        пересчёт предельной эффективности на каждом шаге важен — простое
        пропорциональное деление давало бы переинвестирование в один узел."""
        sens = [r for r in self.sensitivity_analysis() if r['kpi_id'] == kpi_id and r['leverage'] > 0]
        sens.sort(key=lambda r: -r['leverage'])
        candidates = [r['leaf'] for r in sens[:top_k]]
        before = self._kpi_value(kpi_id)
        if not candidates or pool <= 0:
            return {'allocations': {}, 'names': {}, 'kpi_before': before, 'kpi_after': before}
        base_F = {l: float(self.G.nodes[l].get('F', 0.0)) for l in candidates}
        alloc = dict(base_F)
        inc = pool / max(1, steps)
        for _ in range(steps):
            cur = self._simulate_assignment(alloc)[kpi_id]
            best, best_gain = None, 1e-12
            for l in candidates:
                trial = dict(alloc); trial[l] += inc
                gain = self._simulate_assignment(trial)[kpi_id] - cur
                if gain > best_gain:
                    best_gain, best = gain, l
            if best is None:
                break
            alloc[best] += inc
        after = self._simulate_assignment(alloc)[kpi_id]
        allocations = {l: round(alloc[l] - base_F[l], 3) for l in candidates if alloc[l] - base_F[l] > 1e-6}
        names = {l: self.G.nodes[l].get('name', l) for l in allocations}
        # ДОБАВЛЕНО (B3): вклад каждой работы в прирост KPI (сколько KPI потеряет, если убрать
        # её догрузку) — чтобы показывать ИЗМЕНЕНИЕ KPI по каждой работе, а не только сумму.
        deltas = {}
        for l in allocations:
            trial = dict(alloc); trial[l] = base_F[l]
            deltas[l] = round(after - self._simulate_assignment(trial)[kpi_id], 4)
        return {'allocations': allocations, 'names': names, 'deltas': deltas,
                'kpi_before': before, 'kpi_after': after}

    def _remap_window(self, p_old, p_new, c_start, c_end):
        """Линейно переносит окно ребёнка из СТАРОГО окна родителя в НОВОЕ: сохраняет
        относительное положение начала и масштабирует длительность. Тождественно, если
        окно родителя не изменилось (тогда даты ребёнка не двигаются)."""
        try:
            po_s, po_e = self._pdate(p_old[0]), self._pdate(p_old[1])
            pn_s, pn_e = self._pdate(p_new[0]), self._pdate(p_new[1])
            cs, ce = self._pdate(c_start), self._pdate(c_end)
        except Exception:
            return c_start, c_end
        old_span = max(1, (po_e - po_s).days)
        new_span = max(1, (pn_e - pn_s).days)
        scale = new_span / old_span
        start_off = (cs - po_s).days
        c_days = max(0, (ce - cs).days)
        new_cs = pn_s + timedelta(days=int(round(start_off * scale)))
        new_ce = new_cs + timedelta(days=int(round(c_days * scale)))
        return new_cs.strftime("%Y-%m-%d"), new_ce.strftime("%Y-%m-%d")

    def _leaf_descendants(self, node: str) -> List[str]:
        """Все листовые работы в поддереве узла (его дети, внуки … без своих подзадач)."""
        return [n for n in nx.ancestors(self.G, node)
                if str(self.G.nodes[n].get('type', '')).upper() != 'KPI' and self.G.in_degree(n) == 0]

    def display_finances(self, node_id: str) -> Dict[str, Dict[str, float]]:
        """ЭФФЕКТИВНЫЕ финансы узла по годам {год: {base, req_extra, add}} — то, что реально
        видит пользователь: у листа его собственные+распределённые деньги; у родителя — сумма
        по листьям поддерева. ЕДИНЫЙ источник «текущего» состояния для таблицы, живого прогноза
        и сравнения «было/стало» (нельзя брать из session_state — там _orig затирается ручной
        правкой ячейки, и «было» ошибочно совпадало со «стало»)."""
        attr = self.G.nodes.get(node_id, {})
        if self.G.in_degree(node_id) == 0:
            return self._parse_finances(attr.get('finances_eff', attr.get('finances', {})))
        agg: Dict[str, Dict[str, float]] = {}
        for lf in self._leaf_descendants(node_id):
            lfin = self._parse_finances(self.G.nodes[lf].get('finances_eff',
                                                             self.G.nodes[lf].get('finances', {})))
            for y_str, amounts in lfin.items():
                acc = agg.setdefault(str(y_str), {'base': 0.0, 'req_extra': 0.0, 'add': 0.0})
                for s in ('base', 'req_extra', 'add'):
                    acc[s] += float(amounts.get(s, 0.0) or 0.0)
        return agg

    def _apply_parent_window_to_descendants(self, parent: str, win_old, win_new):
        """РЕАЛИЗОВАНО (авто-балансировка для родителей)."""
        for leaf in self._leaf_descendants(parent):
            a = self.G.nodes[leaf]
            is_ms = self._is_milestone_type(a.get('type'))
            ns_, ne_ = self._remap_window(win_old, win_new, a.get('T_start'), a.get('T_end'))
            if is_ms: ns_ = ne_
            a['T_start'], a['T_end'] = ns_, ne_
            
            r_req = float(a.get('rho_req', 1.0))
            r_add = float(a.get('rho_add', 0.0))
            # ЭФФЕКТИВНЫЕ финансы листа (с распределёнными от родителей), а не пустой источник
            fin_src = a.get('finances_eff', a.get('finances', {}))
            f_eff, f_real, last_year = self._evaluate_node_finances(fin_src, rho_req=r_req, rho_add=r_add)
            has_fin = (f_eff > 1e-9) or (f_real > 1e-9) or (last_year is not None)
            F_use = f_real if has_fin else 0.0  # деньги только из финансов; нет записи → 0

            if is_ms:
                a['local_value'] = self._milestone_value(leaf, F_use, ne_)
            else:
                d = max(1, (self._pdate(ne_) - self._pdate(ns_)).days)
                a['local_value'] = self._calculate_local_value(F_use, d, T_opt=a.get('T_opt', d), late_days=self._late_days(a, a.get('T_end')))

    def _finances_for_nominal(self, leaf: str, new_F: float) -> dict:
        """Профиль финансов листа, отмасштабированный так, чтобы НОМИНАЛЬНЫЙ бюджет стал new_F.

        Нужен потому, что F листа ВЫЧИСЛЯЕТСЯ из таблицы финансов: попытка просто присвоить
        node['F'] стирается ближайшим пересчётом. Любой «что если бюджет = X» должен идти
        через финансы — иначе аналитика (рычаг, портфель, перераспределение) молча даёт нули."""
        a = self.G.nodes[leaf]
        cur = self._parse_finances(a.get('finances_eff', a.get('finances', {})))
        r_req = float(a.get('rho_req', 1.0))
        r_add = float(a.get('rho_add', 0.0))
        f_eff, _, _ = self._evaluate_node_finances(cur, r_req, r_add)
        new_F = max(0.0, float(new_F))
        if cur and f_eff > 1e-9:
            k = new_F / f_eff
            return {str(y): {s: float(v.get(s, 0.0) or 0.0) * k for s in ('base', 'req_extra', 'add')}
                    for y, v in cur.items() if isinstance(v, dict)}
        # денег не было — кладём базой в ГОД НАЧАЛА работы (чтобы не создавать кассовый разрыв)
        try:
            y0 = self._pdate(a.get('T_start') or a.get('T_end')).year
        except Exception:
            y0 = int(getattr(self.config, 'base_year', 2026))
        return {str(y0): {'base': new_F, 'req_extra': 0.0, 'add': 0.0}}

    def _is_financial_leaf(self, leaf: str) -> bool:
        a = self.G.nodes[leaf]
        if bool(a.get('is_financial', False)):
            return True
        fin = self._parse_finances(a.get('finances_eff', a.get('finances', {})))
        return any(float(v.get(s, 0.0) or 0.0) > 1e-9
                   for v in fin.values() if isinstance(v, dict)
                   for s in ('base', 'req_extra', 'add'))

    def mix(self, entity_id: str, new_F: float, new_start: str, new_end: str, project: bool = True, new_finances: dict = None, rho_req: float = 1.0, rho_add: float = 0.0) -> Dict[str, Dict[str, Any]]:
        kpi_nodes = self.kpi_ids
        win_old = (self.G.nodes[entity_id].get('T_start'), self.G.nodes[entity_id].get('T_end'))
        shares = {kpi: self._entity_kpi_share(entity_id, kpi) for kpi in kpi_nodes} if project else {}

        children = list(self.G.predecessors(entity_id))
        real_children = [c for c in children if str(self.G.nodes[c].get('type','')).upper() != 'KPI']
        is_leaf = len(real_children) == 0

        # «Что если бюджет работы = new_F» БЕЗ явного профиля финансов: F листа вычисляется из
        # таблицы финансов, поэтому простое присвоение F стёрлось бы при первом же пересчёте
        # (именно из-за этого рычаг/портфель/перераспределение молча возвращали нули).
        # Переводим запрошенный номинал в профиль финансов и идём обычным финансовым путём.
        if new_finances is None and is_leaf:
            _cur_F = float(self.G.nodes[entity_id].get('F', 0.0))
            if abs(float(new_F) - _cur_F) > 1e-9:
                new_finances = self._finances_for_nominal(entity_id, float(new_F))
                rho_req = float(self.G.nodes[entity_id].get('rho_req', 1.0))
                rho_add = float(self.G.nodes[entity_id].get('rho_add', 0.0))

        snapshot = {}
        # Снапшотим ВСЕ узлы-работы: финансовая правка очищает ролл-апы предков/промежуточных
        # по всему пути, и пересборка finances_eff затрагивает все листья — восстановление
        # в finally должно вернуть всё. Снимаем ДО любых мутаций (в т.ч. канонической базы).
        nodes_to_snapshot = set(n for n in self.G.nodes()
                                if str(self.G.nodes[n].get('type', '')).upper() != 'KPI')
        for nid in nodes_to_snapshot:
            snapshot[nid] = {
                'F': self.G.nodes[nid].get('F', 0.0),
                'T_start': self.G.nodes[nid].get('T_start', ''),
                'T_end': self.G.nodes[nid].get('T_end', ''),
                'local_value': self.G.nodes[nid].get('local_value', 0.0),
                'finances': copy.deepcopy(self.G.nodes[nid].get('finances', {})),
                'finances_eff': copy.deepcopy(self.G.nodes[nid].get('finances_eff', None)),
                'rho_req': self.G.nodes[nid].get('rho_req', 1.0),
                'rho_add': self.G.nodes[nid].get('rho_add', 0.0),
                'is_financial': self.G.nodes[nid].get('is_financial', False)
            }

        # КАНОНИЧЕСКАЯ БАЗА для сравнения. Правка финансов узла очищает ролл-апы предков и
        # перераспределяет деньги — это меняет режим распределения. Если снять «было» из старого
        # (нетронутого) режима, а «стало» из нового, получим ФАНТОМНОЕ изменение KPI даже когда
        # деньги/сроки не менялись (сосед-лист терял распределённую от родителя долю). Поэтому
        # «было» снимаем ИЗ ТОГО ЖЕ конвейера: прогоняем ТЕКУЩИЙ источник узла через очистку
        # ролл-апов + перераспределение, и только затем фиксируем old_kpis. Тогда «ничего не
        # меняли» → old == new → m = 1.0 → прогноз не дёргается.
        if new_finances is not None:
            # База = то, что РЕАЛЬНО показывает матрица узла (его эффективные деньги — свои
            # плюс распределённые от родителя). Прогоняем эту базу через тот же конвейер, что и
            # правку, и только затем снимаем old_kpis. Тогда «стало == матрице» ⇒ pct = 0.
            _is_leaf_e = self.G.in_degree(entity_id) == 0
            if _is_leaf_e:
                _base_src = self._parse_finances(
                    self.G.nodes[entity_id].get('finances_eff',
                                                 self.G.nodes[entity_id].get('finances', {})))
                self.G.nodes[entity_id]['finances'] = copy.deepcopy(_base_src)
            self._clear_rollup_sources(entity_id)
            self._compute_effective_finances()
            self._recompute_leaf_values()
            self._recompute_parent_budgets()
            self._propagate_all_kpis()
            old_kpis = {kpi: self._kpi_value(kpi) for kpi in kpi_nodes}
        else:
            old_kpis = {kpi: self._kpi_value(kpi) for kpi in kpi_nodes}

        try:
            self.G.nodes[entity_id]['rho_req'] = rho_req
            self.G.nodes[entity_id]['rho_add'] = rho_add
            if not is_leaf:
                descendants = self._leaf_descendants(entity_id)
                if real_children:
                    descendants.extend(real_children)
                for child in set(descendants):
                    if child in self.G.nodes:
                        self.G.nodes[child]['rho_req'] = rho_req
                        self.G.nodes[child]['rho_add'] = rho_add

            if new_finances is not None:
                self.G.nodes[entity_id]['finances'] = copy.deepcopy(new_finances)
                # ввод денег в нефинансовую сущность автоматически делает её финансовой
                if any(float(v.get(st_, 0.0) or 0.0) > 1e-9 for v in (new_finances or {}).values()
                       if isinstance(v, dict) for st_ in ('base', 'req_extra', 'add')):
                    self.G.nodes[entity_id]['is_financial'] = True
                self._clear_rollup_sources(entity_id)  # предпросмотр: снапшот восстановит в finally
                f_eff, f_real, last_year = self._evaluate_node_finances(new_finances, rho_req, rho_add)
                new_F = f_eff 
                if is_leaf:
                    # согласовать eff остальных листьев с очисткой ролл-апов и новой правкой
                    self._compute_effective_finances()
                    self._recompute_leaf_values()
                    self._recompute_parent_budgets()
            else:
                f_eff, f_real, last_year = self._evaluate_node_finances(self.G.nodes[entity_id].get('finances', {}), rho_req, rho_add)
                if float(self.G.nodes[entity_id].get('F', 0.0)) > 1e-9 and f_real <= 1e-9:
                    f_real = float(self.G.nodes[entity_id].get('F', 0.0))

            if is_leaf:
                is_ms = self._is_milestone_type(self.G.nodes[entity_id].get('type'))
                if is_ms: new_start = new_end
                self.G.nodes[entity_id].update({'F': float(new_F), 'T_start': new_start, 'T_end': new_end})
                
                use_f_real = f_real if (f_real > 1e-9 or new_F <= 1e-9) else float(new_F)

                if is_ms:
                    self.G.nodes[entity_id]['local_value'] = self._milestone_value(entity_id, use_f_real, new_end)
                else:
                    delta_days = max(1, (datetime.strptime(new_end, "%Y-%m-%d") - datetime.strptime(new_start, "%Y-%m-%d")).days)
                    self.G.nodes[entity_id]['local_value'] = self._calculate_local_value(
                        use_f_real, delta_days, T_opt=self.G.nodes[entity_id]['T_opt'],
                        late_days=self._late_days(self.G.nodes[entity_id], new_end))
            else:
                if new_finances is not None:
                    # ФИНАНСОВЫЙ сценарий на родителе: обновляем источник и пересобираем
                    # эффективные финансы листьев + их ценность ТОЙ ЖЕ логикой, что и на базе
                    # (консистентно, симметрично, с учётом rho). Даёт корректный знак и величину.
                    self.G.nodes[entity_id]['finances'] = copy.deepcopy(new_finances)
                    self._compute_effective_finances()
                    self._recompute_leaf_values()
                    self._recompute_parent_budgets()
                elif real_children:
                    # БЮДЖЕТНЫЙ сценарий (без финансов) — водопадное распределение по весам.
                    queue = [(entity_id, new_F, float(self.G.nodes[entity_id].get('F', 0.0)), self.G.nodes[entity_id].get('finances', {}))]
                    while queue:
                        curr_id, c_new_F, c_old_F, c_fin = queue.pop(0)
                        c_children = [c for c in self.G.predecessors(curr_id) if str(self.G.nodes[c].get('type','')).upper() != 'KPI']
                        if not c_children:
                            continue
                        dist = self._child_distribution_weights(curr_id, c_children)
                        old_F_map = {c: float(self.G.nodes[c].get('F', 0.0)) for c in c_children}
                        delta_F = float(c_new_F) - c_old_F
                        new_F_map = self._waterfall_distribute(c_children, dist, old_F_map, delta_F)
                        for child in c_children:
                            self.G.nodes[child]['F'] = new_F_map[child]
                            queue.append((child, new_F_map[child], old_F_map[child], self.G.nodes[child].get('finances', {})))
                    self._recompute_leaf_values()
                    self._recompute_parent_budgets()

                self.G.nodes[entity_id].update({'F': float(new_F), 'T_start': new_start, 'T_end': new_end})
                self.G.nodes[entity_id]['local_value'] = 0.0
                self._apply_parent_window_to_descendants(entity_id, win_old, (new_start, new_end))

            self._propagate_all_kpis()

            output = {}
            for kpi in kpi_nodes:
                node_attr = self.G.nodes[kpi]
                old_v = old_kpis[kpi]
                new_v = self._kpi_value(kpi)

                def safe_float(val):
                    try: return float(val)
                    except (ValueError, TypeError): return 0.0

                m_raw = (new_v / old_v) if old_v > 0 else 1.0
                m = self._calibrate_m(kpi, m_raw)
                pct_change = m - 1.0
                if project:
                    # ИСПРАВЛЕННАЯ ЛОГИКА: ищем самые глубокие работы (листья), 
                    # чья ценность РЕАЛЬНО изменилась (пострадала от среза бюджета или выросла)
                    _changed_ends = []
                    _check_nodes = self._leaf_descendants(entity_id) if not is_leaf else [entity_id]
                    
                    for cn in _check_nodes:
                        _old_v = snapshot.get(cn, {}).get('local_value', 0.0)
                        _new_v = self.G.nodes[cn].get('local_value', 0.0)
                        # Если работа реально потеряла деньги и её ценность упала
                        if abs(_new_v - _old_v) > 1e-9:
                            _ce = self.G.nodes[cn].get('T_end')
                            if _ce:
                                _changed_ends.append(_ce)
                                
                    # Если ничего не изменилось или это просто одиночная задача
                    if not _changed_ends:
                        _ce = self.G.nodes[entity_id].get('T_end')
                        if _ce:
                            _changed_ends.append(_ce)

                    periods_out, annual = self.project_periods(win_old, (new_start, new_end), kpi, m, shares.get(kpi, 0.0), children_ends=_changed_ends)
                    for _p in periods_out: _p['forecast'] = self._clamp_kpi_forecast(kpi, _p.get('forecast', 0.0))
                    for _y in annual: annual[_y]['forecast'] = self._clamp_kpi_forecast(kpi, annual[_y].get('forecast', 0.0))
                else:
                    periods_out, annual = [], {}

                q_label = {1: 'I квартал', 2: 'II квартал', 3: 'III квартал', 4: 'IV квартал'}
                quarters = {}
                if periods_out:
                    base_year = min(r['year'] for r in periods_out)
                    for r in periods_out:
                        if r['year'] == base_year:
                            quarters[q_label[r['q']]] = {'plan': r['plan'], 'forecast': r['forecast']}
                    ay = annual.get(base_year, {})
                    quarters['Год'] = {'plan': ay.get('plan', safe_float(node_attr.get('Year', 0))),
                                       'forecast': self._clamp_kpi_forecast(kpi, ay.get('forecast', safe_float(node_attr.get('Year', 0)) * m))}
                else:
                    yv = safe_float(node_attr.get('Year', 0))
                    for k_, lbl in (('Q1', 'I квартал'), ('Q2', 'II квартал'), ('Q3', 'III квартал'), ('Q4', 'IV квартал')):
                        pv = safe_float(node_attr.get(k_, 0))
                        quarters[lbl] = {'plan': pv, 'forecast': self._clamp_kpi_forecast(kpi, pv * m)}
                    quarters['Год'] = {'plan': yv, 'forecast': self._clamp_kpi_forecast(kpi, yv * m)}

                output[kpi] = {
                    'old': old_v, 'new': new_v, 'delta': new_v - old_v, 'pct_change': pct_change,
                    'share': shares.get(kpi, 0.0),
                    'periods': periods_out, 'annual': annual,
                    'quarters': quarters,
                }
            return output

        finally:
            for nid, state in snapshot.items():
                self.G.nodes[nid].update(state)
            self._propagate_all_kpis()

    def commit(self, entity_id: str, new_F: float, new_start: str, new_end: str, simulation_results: Dict[str, Any], new_finances: dict = None, rho_req: float = 1.0, rho_add: float = 0.0):
        children = list(self.G.predecessors(entity_id))
        real_children = [c for c in children if str(self.G.nodes[c].get('type','')).upper() != 'KPI']
        is_leaf = len(real_children) == 0

        self.G.nodes[entity_id]['rho_req'] = rho_req
        self.G.nodes[entity_id]['rho_add'] = rho_add
        if not is_leaf:
            descendants = self._leaf_descendants(entity_id)
            if real_children:
                descendants.extend(real_children)
            for child in set(descendants):
                if child in self.G.nodes:
                    self.G.nodes[child]['rho_req'] = rho_req
                    self.G.nodes[child]['rho_add'] = rho_add

        if new_finances is not None:
            # НЕ трогаем распределение, если деньги фактически не изменились (повторное применение
            # того же профиля): иначе очистка ролл-апов превратила бы распределённые от родителя
            # деньги в собственные и обнулила бы соседей. Меняем источник/чистим ролл-апы только
            # при реальной правке денег.
            _eff_now = self.G.nodes[entity_id].get('finances_eff', self.G.nodes[entity_id].get('finances', {}))
            _real_change = not self._finances_match(new_finances, _eff_now)
            if _real_change:
                self.G.nodes[entity_id]['finances'] = copy.deepcopy(new_finances)
                if any(float(v.get(st_, 0.0) or 0.0) > 1e-9 for v in (new_finances or {}).values()
                       if isinstance(v, dict) for st_ in ('base', 'req_extra', 'add')):
                    self.G.nodes[entity_id]['is_financial'] = True
                self._clear_rollup_sources(entity_id)  # старые ролл-апы предков/промежуточных устарели
            f_eff, f_real, last_year = self._evaluate_node_finances(new_finances, rho_req, rho_add)
            new_F = f_eff
        else:
            f_eff, f_real, last_year = self._evaluate_node_finances(self.G.nodes[entity_id].get('finances', {}), rho_req, rho_add)
            if float(self.G.nodes[entity_id].get('F', 0.0)) > 1e-9 and f_real <= 1e-9:
                f_real = float(self.G.nodes[entity_id].get('F', 0.0))

        if is_leaf:
            is_ms = self._is_milestone_type(self.G.nodes[entity_id].get('type'))
            if is_ms:
                new_start = new_end
            self.G.nodes[entity_id].update({'F': float(new_F), 'T_start': new_start, 'T_end': new_end})

            use_f_real = f_real if (f_real > 1e-9 or new_F <= 1e-9) else float(new_F)

            if is_ms:
                self.G.nodes[entity_id]['local_value'] = self._milestone_value(entity_id, use_f_real, new_end)
            else:
                delta_days = max(1, (datetime.strptime(new_end, "%Y-%m-%d") - datetime.strptime(new_start, "%Y-%m-%d")).days)
                self.G.nodes[entity_id]['local_value'] = self._calculate_local_value(
                    use_f_real, delta_days, T_opt=self.G.nodes[entity_id]['T_opt'],
                    late_days=self._late_days(self.G.nodes[entity_id], new_end))
        else:
            win_old_parent = (self.G.nodes[entity_id].get('T_start'), self.G.nodes[entity_id].get('T_end'))
            if new_finances is not None:
                # ФИНАНСОВЫЙ сценарий на родителе: та же логика, что и в mix() — ПЕРЕСОБРАТЬ
                # эффективные финансы листьев из источника (не «заморожено» с прошлой загрузки).
                self._compute_effective_finances()
                self._recompute_leaf_values()
                self._recompute_parent_budgets()
            elif real_children:
                # БЮДЖЕТНЫЙ сценарий (без финансов) — водопадное распределение по весам.
                queue = [(entity_id, new_F, float(self.G.nodes[entity_id].get('F', 0.0)), self.G.nodes[entity_id].get('finances', {}))]
                while queue:
                    curr_id, c_new_F, c_old_F, c_fin = queue.pop(0)
                    c_children = [c for c in self.G.predecessors(curr_id) if str(self.G.nodes[c].get('type','')).upper() != 'KPI']
                    if not c_children:
                        continue
                    dist = self._child_distribution_weights(curr_id, c_children)
                    old_F_map = {c: float(self.G.nodes[c].get('F', 0.0)) for c in c_children}
                    delta_F = float(c_new_F) - c_old_F
                    new_F_map = self._waterfall_distribute(c_children, dist, old_F_map, delta_F)
                    for child in c_children:
                        self.G.nodes[child]['F'] = new_F_map[child]
                        queue.append((child, new_F_map[child], old_F_map[child], self.G.nodes[child].get('finances', {})))
                self._recompute_leaf_values()
                self._recompute_parent_budgets()

            self.G.nodes[entity_id].update({'F': float(new_F), 'T_start': new_start, 'T_end': new_end})
            self.G.nodes[entity_id]['local_value'] = 0.0
            self._apply_parent_window_to_descendants(entity_id, win_old_parent, (new_start, new_end))

        # Каскад срыва срока по зависимостям предшествования (если заданы).
        self._precedence_cascade(entity_id)

        # ГАРАНТИРОВАННАЯ согласованность после любой финансовой правки (лист или родитель):
        # пересобрать эффективные финансы из источника, пересчитать листья и бюджеты родителей.
        if new_finances is not None:
            self._compute_effective_finances()
            self._recompute_leaf_values()
            self._recompute_parent_budgets()

        self._propagate_all_kpis()

        for kpi_id, data in simulation_results.items():
            if kpi_id not in self.G.nodes:
                continue
            periods_out = data.get('periods') or []
            if periods_out:
                stored = {(p['year'], p['q']): p for p in self._kpi_periods(kpi_id)}
                for r in periods_out:
                    key = (r['year'], r['q'])
                    cell = stored.setdefault(key, {'year': r['year'], 'q': r['q'], 'plan': 0.0, 'forecast': 0.0, 'fact': 0.0})
                    if not r.get('locked'):
                        cell['plan'] = r['forecast']
                        cell['forecast'] = r['forecast']
                self.G.nodes[kpi_id]['periods'] = json.dumps(list(stored.values()), ensure_ascii=False)

            annual_data = data.get('annual') or {}
            if annual_data:
                cur_annual = self._kpi_annual(kpi_id)
                for yr, av in annual_data.items():
                    yr = int(yr)
                    cell = cur_annual.setdefault(yr, {'plan': 0.0, 'forecast': 0.0, 'fact': 0.0})
                    cell['plan'] = av.get('forecast', cell.get('plan', 0.0))
                    cell['forecast'] = av.get('forecast', cell.get('forecast', 0.0))
                self.G.nodes[kpi_id]['annual'] = json.dumps({str(y): v for y, v in cur_annual.items()}, ensure_ascii=False)
                base_year = min(cur_annual) if cur_annual else None
                if base_year is not None:
                    self.G.nodes[kpi_id]['Year'] = cur_annual[base_year].get('plan', self.G.nodes[kpi_id].get('Year'))

            quarters = data.get('quarters', {})
            if 'I квартал' in quarters: self.G.nodes[kpi_id]['Q1'] = quarters['I квартал']['forecast']
            if 'II квартал' in quarters: self.G.nodes[kpi_id]['Q2'] = quarters['II квартал']['forecast']
            if 'III квартал' in quarters: self.G.nodes[kpi_id]['Q3'] = quarters['III квартал']['forecast']
            if 'IV квартал' in quarters: self.G.nodes[kpi_id]['Q4'] = quarters['IV квартал']['forecast']
            if 'Год' in quarters: self.G.nodes[kpi_id]['Year'] = quarters['Год']['forecast']

        self._compute_quarter_windows()
        self._validate_budgets()
        self._validate_schedule()

        self._validate_schedule()