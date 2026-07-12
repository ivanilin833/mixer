# project_store.py — управление несколькими проектами «Микшера».
#
# Каждый проект — это папка projects/<slug>/ со своими план-графиком, показателями,
# методиками, кешами (веса связей, сжатые методики) и снапшотами (утверждённый
# базовый план + сохранённые сценарии). Реестр не нужен: проекты обнаруживаются
# автоматически по наличию schedule.xlsx.

import os
import re
import io
import json
import time
import sys
import uuid
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any


def _atomic_write_bytes(path: str, data: bytes):
    """Атомарная запись бинарного файла (уникальный temp + os.replace).

    Уникальное имя temp (pid+uuid) исключает коллизии между потоками сборки;
    fsync гарантирует, что данные на диске, а не только в буфере ОС."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_write_json(path: str, payload: Any):
    """Атомарная запись JSON (через _atomic_write_bytes)."""
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
    _atomic_write_bytes(path, data)


def _default_projects_root() -> str:
    """Папка с проектами.

    В собранном приложении (PyInstaller / streamlit-desktop-app) sys.frozen == True —
    тогда кладём данные РЯДОМ С EXE (папка, где лежит исполняемый файл), чтобы они
    сохранялись между запусками и не терялись во временной распаковке onefile.
    В обычном запуске из исходников — относительная папка 'projects' (как раньше)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        return os.path.join(base, "projects")
    # переопределение через переменную окружения (удобно для тестов/переноса)
    env = os.environ.get("MIXER_PROJECTS_ROOT")
    return env if env else "projects"


PROJECTS_ROOT = _default_projects_root()

SCHEDULE_NAME = "schedule.xlsx"
INDICATORS_NAME = "indicators.xlsx"
FINANCES_NAME = "finances.xlsx"
METHODOLOGIES_DIR = "methodologies"
MANIFEST_NAME = "project.json"

# ----- Глобальные настройки подключения к ИИ (одни на приложение) -----
LLM_SETTINGS_NAME = "llm_settings.json"

# Пресеты провайдеров: подставляют base_url/модель, остальное правит пользователь.
LLM_PRESETS = {
    "local":  {"label": "Локальная (Ollama)",            "base_url": "http://localhost:11434/v1", "model": "gpt-oss:20b", "needs_key": False},
    "openai": {"label": "OpenAI API",                     "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "needs_key": True},
    "custom": {"label": "OpenAI-совместимый (свой URL)",  "base_url": "",                          "model": "",            "needs_key": True},
}


def default_llm_settings() -> Dict[str, Any]:
    return {"enabled": True, "provider": "local",
            "base_url": "http://localhost:11434/v1", "api_key": "",
            "model": "gpt-oss:20b", "timeout": 120}


def llm_settings_path(root_dir: str = PROJECTS_ROOT) -> str:
    return os.path.join(root_dir, LLM_SETTINGS_NAME)


def load_llm_settings(root_dir: str = PROJECTS_ROOT) -> Dict[str, Any]:
    d = default_llm_settings()
    try:
        p = llm_settings_path(root_dir)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                d.update({k: v for k, v in (json.load(f) or {}).items() if k in d})
    except Exception:
        pass
    return d


def save_llm_settings(settings: Dict[str, Any], root_dir: str = PROJECTS_ROOT):
    os.makedirs(root_dir, exist_ok=True)
    base = default_llm_settings()
    base.update({k: settings.get(k, base[k]) for k in base})
    _atomic_write_json(llm_settings_path(root_dir), base)
    return base



# ======================================================================
# КОНТЕКСТ ПРОЕКТА
# ======================================================================
@dataclass
class ProjectContext:
    slug: str
    title: str
    root: str

    @property
    def schedule_path(self) -> str:
        return os.path.join(self.root, SCHEDULE_NAME)

    @property
    def indicators_path(self) -> str:
        return os.path.join(self.root, INDICATORS_NAME)

    @property
    def finances_path(self) -> str: # <--- НОВОЕ
        return os.path.join(self.root, FINANCES_NAME)

    def has_finances(self) -> bool: # <--- НОВОЕ
        return os.path.exists(self.finances_path)

    @property
    def methodologies_dir(self) -> str:
        return os.path.join(self.root, METHODOLOGIES_DIR)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.root, ".cache")

    @property
    def weights_path(self) -> str:
        return os.path.join(self.cache_dir, "weights_matrix.json")

    @property
    def meth_cache_dir(self) -> str:
        return os.path.join(self.cache_dir, "methodologies")

    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.root, "snapshots")

    @property
    def baseline_path(self) -> str:
        return os.path.join(self.snapshots_dir, "baseline.json")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    @property
    def settings_path(self) -> str:
        return os.path.join(self.root, "settings.json")

    @property
    def log_path(self) -> str:
        """Отдельный лог-файл проекта (все его сообщения; ошибки дублируются в общий)."""
        return os.path.join(self.root, "project.log")

    def ensure_dirs(self):
        for d in (self.root, self.cache_dir, self.meth_cache_dir, self.snapshots_dir, self.methodologies_dir):
            os.makedirs(d, exist_ok=True)

    # --- метаданные / состояние файлов ---
    def has_schedule(self) -> bool:
        return os.path.exists(self.schedule_path)

    def has_indicators(self) -> bool:
        return os.path.exists(self.indicators_path)

    def methodology_files(self) -> List[str]:
        d = self.methodologies_dir
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if not f.startswith('.'))

    def has_baseline(self) -> bool:
        return os.path.exists(self.baseline_path)

    def file_signature(self) -> tuple:
        """Подпись для инвалидации кеша: размеры/время изменения ключевых файлов."""
        def sig(p):
            try:
                s = os.stat(p)
                return (round(s.st_mtime, 3), s.st_size)
            except OSError:
                return (0.0, 0)
        # По каждому файлу методик берём (имя, mtime, size): mtime каталога на NTFS не
        # меняется при перезаписи содержимого существующего файла, поэтому одного mtime
        # папки недостаточно — кеш сжатых методик/весов не инвалидировался бы.
        meth_sig = tuple()
        if os.path.isdir(self.methodologies_dir):
            try:
                names = sorted(f for f in os.listdir(self.methodologies_dir) if not f.startswith('.'))
                meth_sig = tuple((n,) + sig(os.path.join(self.methodologies_dir, n)) for n in names)
            except OSError:
                meth_sig = tuple()
        base_sig = sig(self.baseline_path)  # утверждённый план тоже влияет на состояние
        return (sig(self.schedule_path), sig(self.indicators_path), sig(self.finances_path), meth_sig, base_sig)

    def last_modified(self) -> Optional[float]:
        times = []
        for p in (self.schedule_path, self.indicators_path, self.finances_path, self.baseline_path):
            try:
                times.append(os.stat(p).st_mtime)
            except OSError:
                continue
        return max(times) if times else None
    


# ======================================================================
# РЕЕСТР (автообнаружение)
# ======================================================================
def slugify(title: str) -> str:
    """Безопасное имя папки из названия проекта (латиница/цифры/дефис)."""
    translit = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    s = (title or "").strip().lower()
    s = "".join(translit.get(ch, ch) for ch in s)
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    if not s:
        return f"project-{uuid.uuid4().hex[:8]}"
    # Зарезервированные имена устройств Windows (CON, PRN, COM1 …) нельзя использовать
    # как имя папки — префиксуем, чтобы создание/открытие не ломалось.
    _reserved = {'con', 'prn', 'aux', 'nul'} | {f'com{i}' for i in range(1, 10)} | {f'lpt{i}' for i in range(1, 10)}
    if s in _reserved:
        s = f"p-{s}"
    return s


def _read_manifest(root: str) -> Dict[str, Any]:
    p = os.path.join(root, MANIFEST_NAME)
    if os.path.exists(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def context_for(slug: str, root_dir: str = PROJECTS_ROOT) -> ProjectContext:
    # Защита от выхода за корень проектов: slug — это ИМЯ ПАПКИ, а не путь.
    safe = os.path.basename(str(slug or "").strip().replace("\\", "/").rstrip("/"))
    if safe in ("", ".", ".."):
        raise ValueError(f"Недопустимое имя проекта: {slug!r}")
    root = os.path.join(root_dir, safe)
    man = _read_manifest(root)
    title = man.get('title') or safe
    return ProjectContext(slug=safe, title=title, root=root)


def discover(root_dir: str = PROJECTS_ROOT) -> List[ProjectContext]:
    """Все подпапки root_dir, где есть schedule.xlsx, — это проекты."""
    if not os.path.isdir(root_dir):
        return []
    out = []
    for name in sorted(os.listdir(root_dir)):
        root = os.path.join(root_dir, name)
        if not os.path.isdir(root) or name.startswith('.'):
            continue
        if os.path.exists(os.path.join(root, SCHEDULE_NAME)):
            out.append(context_for(name, root_dir))
    return out


def create_project(title: str, root_dir: str = PROJECTS_ROOT, slug: Optional[str] = None) -> ProjectContext:
    slug = slug or slugify(title)
    # избегаем коллизий имён папок; создаём каталог атомарно (os.mkdir) и ловим
    # FileExistsError — это исключает гонку двух потоков с одинаковым title.
    base, n = slug, 2
    os.makedirs(root_dir, exist_ok=True)
    while True:
        try:
            os.mkdir(os.path.join(root_dir, slug))
            break
        except FileExistsError:
            slug = f"{base}-{n}"; n += 1
    ctx = context_for(slug, root_dir)
    ctx.ensure_dirs()
    _atomic_write_json(ctx.manifest_path,
                       {'title': title, 'slug': slug, 'created': datetime.now().isoformat(timespec='seconds')})
    return ctx


def save_schedule(ctx: ProjectContext, data: bytes):
    ctx.ensure_dirs()
    _atomic_write_bytes(ctx.schedule_path, data)


def save_indicators(ctx: ProjectContext, data: bytes):
    ctx.ensure_dirs()
    _atomic_write_bytes(ctx.indicators_path, data)


def save_methodology(ctx: ProjectContext, filename: str, data: bytes):
    ctx.ensure_dirs()
    safe = os.path.basename(filename)
    _atomic_write_bytes(os.path.join(ctx.methodologies_dir, safe), data)

def save_finances(ctx: ProjectContext, data: bytes):
    ctx.ensure_dirs()
    _atomic_write_bytes(ctx.finances_path, data)

def delete_project(ctx: ProjectContext):
    if os.path.isdir(ctx.root):
        shutil.rmtree(ctx.root)


# ======================================================================
# НАСТРОЙКИ ПРОЕКТА (единица бюджета и пр.) — персист на диск
# ======================================================================
# Допустимые ключи и значения по умолчанию (расширяемо).
_PROJECT_SETTINGS_DEFAULT = {'budget_scale': 'millions'}


def load_project_settings(ctx: ProjectContext) -> Dict[str, Any]:
    d = dict(_PROJECT_SETTINGS_DEFAULT)
    try:
        if os.path.exists(ctx.settings_path):
            with open(ctx.settings_path, 'r', encoding='utf-8') as f:
                saved = json.load(f) or {}
            d.update({k: v for k, v in saved.items() if k in d})
    except Exception:
        pass
    return d


def save_project_settings(ctx: ProjectContext, settings: Dict[str, Any]):
    ctx.ensure_dirs()
    base = load_project_settings(ctx)
    base.update({k: settings[k] for k in settings if k in _PROJECT_SETTINGS_DEFAULT})
    _atomic_write_json(ctx.settings_path, base)


# ======================================================================
# СНАПШОТЫ: утверждённый базовый план и сохранённые сценарии
# ======================================================================
def save_baseline(ctx: ProjectContext, state: Dict[str, Any]):
    ctx.ensure_dirs()
    payload = dict(state or {})
    payload['saved_at'] = datetime.now().isoformat(timespec='seconds')
    _atomic_write_json(ctx.baseline_path, payload)


def load_baseline(ctx: ProjectContext) -> Optional[Dict[str, Any]]:
    if not ctx.has_baseline():
        return None
    try:
        with open(ctx.baseline_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # Файл ЕСТЬ, но не читается — это повреждение, а не «плана нет».
        # Не возвращаем None молча (иначе UI решит, что baseline отсутствует, и
        # пользователь перезапишет утверждённый план). Отводим битый файл в сторону
        # и поднимаем ошибку, чтобы вызвавший код показал её, а не затёр данные.
        corrupt = ctx.baseline_path + ".corrupt"
        try:
            os.replace(ctx.baseline_path, corrupt)
        except OSError:
            pass
        raise ValueError(f"Повреждён baseline.json ({e}); перемещён в {os.path.basename(corrupt)}") from e


def reset_baseline(ctx: ProjectContext):
    if ctx.has_baseline():
        os.remove(ctx.baseline_path)


def _scenario_path(ctx: ProjectContext, name: str) -> str:
    # Читаемая часть + хеш ПОЛНОГО имени: одинаковое имя → тот же файл (перезапись как
    # обновление), но разные имена ("План/2" и "План2") больше не схлопываются в один файл.
    safe = re.sub(r'[^a-zA-Z0-9а-яА-Я _-]+', '', name).strip()
    digest = hashlib.sha1((name or "").encode('utf-8')).hexdigest()[:10]
    stem = f"{safe}_{digest}" if safe else digest
    return os.path.join(ctx.snapshots_dir, f"scenario_{stem}.json")


def save_scenario(ctx: ProjectContext, name: str, state: Dict[str, Any], meta: Optional[Dict] = None):
    ctx.ensure_dirs()
    payload = {'name': name, 'saved_at': datetime.now().isoformat(timespec='seconds'),
               'meta': meta or {}, 'state': state}
    _atomic_write_json(_scenario_path(ctx, name), payload)


def list_scenarios(ctx: ProjectContext) -> List[Dict[str, Any]]:
    d = ctx.snapshots_dir
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith('scenario_') and fn.endswith('.json'):
            try:
                with open(os.path.join(d, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                out.append({'file': fn, 'name': data.get('name', fn),
                            'saved_at': data.get('saved_at', ''), 'meta': data.get('meta', {}),
                            'state': data.get('state', {})})
            except Exception:
                continue
    return out


def delete_scenario(ctx: ProjectContext, file_name: str):
    p = os.path.join(ctx.snapshots_dir, os.path.basename(file_name))
    if os.path.exists(p):
        os.remove(p)


# ======================================================================
# МИГРАЦИЯ ЛЕГАСИ data/ → projects/<slug>/
# ======================================================================
def migrate_legacy(legacy_dir: str = "data", root_dir: str = PROJECTS_ROOT,
                   title: str = "Проект по умолчанию") -> Optional[ProjectContext]:
    """Однократно переносит старую плоскую раскладку data/ в projects/<slug>/.

    Срабатывает только если в projects/ ещё нет ни одного проекта, а в data/ лежит
    план-график. Файлы копируются (оригинал не трогаем)."""
    if discover(root_dir):
        return None
    legacy_sched = os.path.join(legacy_dir, "plan_grafik.xlsx")
    if not os.path.exists(legacy_sched):
        return None
    ctx = create_project(title, root_dir)
    shutil.copyfile(legacy_sched, ctx.schedule_path)
    legacy_ind = os.path.join(legacy_dir, "planovye_pokazateli.xlsx")
    if os.path.exists(legacy_ind):
        shutil.copyfile(legacy_ind, ctx.indicators_path)
    legacy_meth = os.path.join(legacy_dir, "methodologies")
    if os.path.isdir(legacy_meth):
        for fn in os.listdir(legacy_meth):
            src = os.path.join(legacy_meth, fn)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(ctx.methodologies_dir, fn))
    return ctx
