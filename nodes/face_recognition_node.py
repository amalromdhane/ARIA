"""
Face Recognition Node — InsightFace ArcFace + Multi-Face Name Gate Queue
=========================================================================
Migrations from dlib version:

  INSIGHTFACE ARCFACE
  - Detection:    RetinaFace (inside InsightFace) — tight face bbox only
  - Recognition:  ArcFace 512-dim embeddings (vs dlib 128-dim)
  - Similarity:   cosine similarity (threshold 0.40)
  - Fallback:     dlib kept if InsightFace not installed

  MULTI-FACE NAME GATE QUEUE
  - _name_queue: list of visitor IDs waiting for a name
  - Robot asks names one by one: "What's your name?" → answer →
    "Nice to meet you X! And what's your name?" → answer → done
  - set_visitor_name() pops the queue and asks for the next unknown

  ALL ORIGINAL OPTIMISATIONS KEPT
  1. Async worker thread
  2. Encoding cache (np.ndarray pre-cast)
  3. CLAHE once per frame
  4. Debounced DB saves (5s)
  5. Absence timer at 1Hz
  6. RetinaFace tight face bbox
  7. YOLO fallback downscaled to 640px
  8. Box cache 3 frames
  9. Direct face-location pass to dlib encoder
"""

import os, pickle, queue, random, re, threading, time, uuid
from collections import Counter

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except ImportError:
    INSIGHT_OK = False
    print("[FR] pip install insightface onnxruntime")

try:
    import face_recognition
    DLIB_OK = True
except ImportError:
    DLIB_OK = False
    if not INSIGHT_OK:
        print("[FR] pip install face-recognition")


# ── Arabic → Latin ────────────────────────────────────────────────────────────

_NAME_MAP = {
    "محمد امين":"Mohamed Amine","محمد أمين":"Mohamed Amine",
    "محمد علي":"Mohamed Ali","محمد":"Mohamed","احمد":"Ahmed",
    "أحمد":"Ahmed","امين":"Amine","أمين":"Amine","علي":"Ali",
    "يوسف":"Youssef","خالد":"Khaled","وسام":"Wissam","رامي":"Rami",
    "كريم":"Karim","سامي":"Sami","وليد":"Walid","هاني":"Hani",
    "طارق":"Tarek","بلال":"Bilel","عمر":"Omar","عمار":"Amar",
    "حسين":"Hussein","حسن":"Hassan","صلاح":"Salah","زياد":"Ziad",
    "ياسين":"Yassine","عماد":"Imad","رياض":"Riadh","منير":"Monir",
    "نزار":"Nizar","فؤاد":"Fouad","مراد":"Mourad",
    "الامين":"Lamine","الأمين":"Lamine",
    "أمال":"Amal","امال":"Amal","أمل":"Amal","امل":"Amal",
    "سارة":"Sara","سارا":"Sara","فاطمة":"Fatima","مريم":"Maryam",
    "نور":"Nour","هند":"Hind","لينا":"Lina","ليلى":"Leila",
    "رنا":"Rana","رانيا":"Rania","دينا":"Dina","نادية":"Nadia",
    "ياسمين":"Yasmin","روان":"Rawan","تسنيم":"Tasnim",
}

_CHAR_MAP = {
    "ا":"a","أ":"a","إ":"i","آ":"aa","ب":"b","ت":"t","ث":"th",
    "ج":"j","ح":"h","خ":"kh","د":"d","ذ":"dh","ر":"r","ز":"z",
    "س":"s","ش":"sh","ص":"s","ض":"d","ط":"t","ظ":"z","ع":"a",
    "غ":"gh","ف":"f","ق":"q","ك":"k","ل":"l","م":"m","ن":"n",
    "ه":"h","و":"w","ي":"i","ى":"a","ة":"a","ء":"","ئ":"i","ؤ":"u",
}


def _to_latin(name: str) -> str:
    s = name.strip()
    if s in _NAME_MAP:
        return _NAME_MAP[s]
    parts = []
    for w in s.split():
        if w in _NAME_MAP:
            parts.append(_NAME_MAP[w])
        else:
            r   = re.sub(r"[\u064b-\u0652]", "", w).replace("\u0644\u0627", "la")
            out = "".join(_CHAR_MAP.get(c, c) for c in r)
            out = re.sub(r"[^a-zA-Z\s\-]", "", out)
            out = re.sub(r"(.)\1{2,}", r"\1\1", out)
            parts.append(out.strip().title())
    return " ".join(p for p in parts if p).title()


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


# ── CLAHE ─────────────────────────────────────────────────────────────────────

def _clahe(frame):
    try:
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        mean_l     = float(l.mean())
        std_l      = float(l.std())
        clip       = max(2.0, min(5.0, 3.0 + (30 - std_l) / 10))
        eq         = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        if mean_l < 80:
            g = 0.5 if mean_l < 50 else 0.7
        elif mean_l > 180:
            g = 1.8 if mean_l > 200 else 1.4
        else:
            g = None
        if g is not None:
            tbl = np.array([((i / 255.0) ** g) * 255
                            for i in range(256)], dtype="uint8")
            eq  = cv2.LUT(eq, tbl)
        return cv2.cvtColor(cv2.merge([eq, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


# ── Track ─────────────────────────────────────────────────────────────────────

class Track:
    VOTE_SIZE   = 6
    VOTE_NEEDED = 3

    def __init__(self, tid: str):
        self.tid          = tid
        self.first_seen   = time.time()
        self.last_seen    = time.time()
        self.last_enc     = None
        self.last_box     = None
        self.votes: list  = []
        self.greeted      = False
        self.goodbye_sent = False
        self.registered   = False

    def touch(self, enc, box):
        self.last_seen    = time.time()
        self.last_enc     = enc
        self.last_box     = box
        self.goodbye_sent = False

    def vote(self, vid, dist):
        self.votes.append((vid, dist))
        if len(self.votes) > self.VOTE_SIZE:
            self.votes.pop(0)

    def confirmed_vid(self):
        if len(self.votes) < self.VOTE_NEEDED:
            return None
        ids = [v for v, _ in self.votes if v is not None]
        if not ids:
            return None
        top, count = Counter(ids).most_common(1)[0]
        return top if count >= self.VOTE_NEEDED else None

    def unknown_confirmed(self) -> bool:
        if len(self.votes) < self.VOTE_NEEDED:
            return False
        return sum(1 for v, _ in self.votes if v is None) >= self.VOTE_NEEDED

    def absent(self, thr: float) -> bool:
        return time.time() - self.last_seen > thr

    def dist_to(self, enc) -> float:
        if self.last_enc is None:
            return 1.0
        try:
            if INSIGHT_OK:
                return 1.0 - float(np.dot(self.last_enc, enc))
            if DLIB_OK:
                return float(face_recognition.face_distance(
                    [self.last_enc], enc)[0])
        except Exception:
            pass
        return 1.0


# ── Node ──────────────────────────────────────────────────────────────────────

class FaceRecognitionNode:

    _DETECT_SIZE      = 640
    _BOX_CACHE_FRAMES = 3
    _INSIGHT_THR      = 0.40   # cosine similarity threshold

    def __init__(self, frame_queue, sound_emotion_queue, state_manager,
                 main_system=None,
                 db_path="visitor_database.pkl",
                 yolo_model="yolov8n-face.pt",
                 db_thr=0.52,
                 track_thr=0.55,
                 absence_thr=10.0,
                 rec_interval=0.8,
                 jitters=2):

        self.frame_queue   = frame_queue
        self.sound_q       = sound_emotion_queue
        self.state_manager = state_manager
        self.main_system   = main_system
        self.running       = False

        self.db_path      = db_path
        self.db_thr       = db_thr
        self.track_thr    = track_thr
        self.absence_thr  = absence_thr
        self.rec_interval = rec_interval
        self.jitters      = jitters

        self._lock     = threading.Lock()
        self.visitors  = self._load_db()
        self._enc_cache: dict[str, np.ndarray] = {}
        self._rebuild_enc_cache()

        self._dirty      = False
        self._last_save  = 0.0
        self._SAVE_EVERY = 5.0

        # ── InsightFace (primary) ────────────────────────────────────────────
        self.insight_app = None
        if INSIGHT_OK:
            try:
                app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                self.insight_app = app
                print("[FR] InsightFace ArcFace ready (512-dim embeddings)")
            except Exception as e:
                print(f"[FR] InsightFace error: {e} — falling back to dlib")

        # ── YOLO (dlib fallback detector) ────────────────────────────────────
        self.yolo = None
        if self.insight_app is None and YOLO_OK:
            try:
                self.yolo = YOLO(yolo_model)
                print(f"[FR] YOLO fallback: {yolo_model}")
            except Exception as e:
                print(f"[FR] YOLO load error: {e}")

        self._last_boxes: list    = []
        self._last_box_frame: int = 0

        # ── Tracking ─────────────────────────────────────────────────────────
        self.tracks: dict[str, Track] = {}
        self._tid_ctr = 0
        self._vid2tid: dict[str, str] = {}
        self._unk_tids: list[str]     = []

        self.current_visitor_id      = None
        self.current_visitor_name    = None

        # ── Multi-face name gate queue ────────────────────────────────────────
        # Each unknown visitor gets a slot. Robot asks one by one.
        self._name_queue: list[str] = []
        self._name_lock  = threading.Lock()

        self.last_rec  = 0.0
        self.frame_ctr = 0
        self._greeted_names: set[str] = set()

        self._speech_lock          = threading.Lock()
        self._last_greeting_speech = 0.0
        self._last_goodbye_speech  = 0.0
        self._speech_gap           = 3.0

        self._rec_q: queue.Queue = queue.Queue(maxsize=1)

        engine = "InsightFace ArcFace" if self.insight_app else "dlib"
        print(f"[FR] Ready — {len(self.visitors)} visitors | engine={engine}")

    # ── Encoding cache ────────────────────────────────────────────────────────

    def _rebuild_enc_cache(self):
        cache = {}
        for vid, v in self.visitors.items():
            raw = v.get("encoding")
            if raw is not None and not v.get("name", "").startswith("Visitor_"):
                try:
                    cache[vid] = np.asarray(raw, dtype=np.float64)
                except Exception:
                    pass
        self._enc_cache = cache

    # ── DB helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        if not name or len(name) < 2:
            return False
        if name.startswith("Visitor_"):
            return False
        if len(name.split()) > 3:
            return False
        if any(c in name for c in ["?", "!", "\\", "/"]):
            return False
        if name.strip().endswith(","):
            return False
        NOT_NAMES = {
            "hi","it","ess","done","think","met","before","what","your",
            "name","you","we","the","and","yes","no","ok","okay","back",
        }
        if name.strip().lower() in NOT_NAMES:
            return False
        alpha = sum(1 for c in name
                    if c.isalpha() or "\u0600" <= c <= "\u06FF")
        if alpha < 2:
            return False
        ar = sum(1 for c in name if "\u0600" <= c <= "\u06FF")
        la = sum(1 for c in name if c.isalpha() and ord(c) < 0x600)
        if ar > 0 and la == 0 and len(name.strip()) < 3:
            return False
        return True

    def _load_db(self) -> dict:
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, "rb") as f:
                    db = pickle.load(f)
                clean = {
                    k: v for k, v in db.items()
                    if not v.get("name", "").startswith("Visitor_")
                    and self._is_valid_name(v.get("name", ""))
                }
                removed = len(db) - len(clean)
                if removed:
                    print(f"[FR] Cleaned {removed} invalid entries from DB")
                print(f"[FR] DB: {[v['name'] for v in clean.values()]}")
                return clean
        except Exception as e:
            print(f"[FR] DB load: {e}")
        return {}

    def _mark_dirty(self):
        self._dirty = True

    def _flush(self):
        if not self._dirty:
            return
        if time.time() - self._last_save < self._SAVE_EVERY:
            return
        self._write_db()

    def _write_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with self._lock:
                snapshot = {k: v for k, v in self.visitors.items()
                            if not v.get("name", "").startswith("Visitor_")}
            with open(self.db_path, "wb") as f:
                pickle.dump(snapshot, f)
            self._dirty     = False
            self._last_save = time.time()
        except Exception as e:
            print(f"[FR] DB write: {e}")

    def _save_db(self):
        self._write_db()

    # ── Display name ──────────────────────────────────────────────────────────

    def _dname(self, vid) -> str | None:
        if vid is None:
            return None
        with self._lock:
            entry = self.visitors.get(vid, {})
        name = entry.get("name", vid)
        if not name or name.startswith("Visitor_"):
            return None
        display = entry.get("name_display")
        if display:
            return display
        if _has_arabic(name):
            latin = _to_latin(name)
            if latin:
                with self._lock:
                    if vid in self.visitors:
                        self.visitors[vid]["name_display"] = latin
                self._mark_dirty()
                return latin
        return name

    # ── Multi-face name gate public API ───────────────────────────────────────

    def is_waiting_for_name(self) -> bool:
        with self._name_lock:
            self._name_queue = [
                v for v in self._name_queue
                if v in self.visitors
                and self.visitors[v].get("name", "").startswith("Visitor_")
            ]
            return len(self._name_queue) > 0

    @property
    def pending_name_visitor_id(self):
        """Backward-compat: first item in name queue."""
        with self._name_lock:
            return self._name_queue[0] if self._name_queue else None

    def set_visitor_name(self, visitor_id_or_name, name=None,
                         name_display=None) -> bool:
        """
        Save name for the front of the queue.
        Automatically asks for the next unknown if queue not empty.
        """
        actual = (name if name is not None else visitor_id_or_name).strip()
        ar = sum(1 for c in actual if "\u0600" <= c <= "\u06FF")
        la = sum(1 for c in actual if c.isalpha() and ord(c) < 0x600)
        if la > ar:
            actual = actual.title()
        if not actual or len(actual) < 2:
            return False

        with self._name_lock:
            if not self._name_queue:
                print("[FR] set_visitor_name: queue empty — ignored")
                return False
            vid       = self._name_queue.pop(0)
            remaining = list(self._name_queue)

        with self._lock:
            if vid not in self.visitors:
                print(f"[FR] set_visitor_name: {vid} not in DB — ignored")
                return False
            self.visitors[vid]["name"] = actual
            if name_display and name_display.strip():
                disp = name_display.strip().title()
            elif ar == 0:
                disp = actual
            else:
                disp = _to_latin(actual) or actual
            self.visitors[vid]["name_display"] = disp
            raw = self.visitors[vid].get("encoding")
            if raw is not None:
                self._enc_cache[vid] = np.asarray(raw, dtype=np.float64)

        self._write_db()
        print(f"[FR] Name saved: '{actual}' display='{disp}' → {vid}")

        tid = self._vid2tid.get(vid)
        if tid and tid in self.tracks:
            self.tracks[tid].greeted = True
        if vid in self._unk_tids:
            self._unk_tids.remove(vid)

        self.current_visitor_id   = None
        self.current_visitor_name = actual

        ai = getattr(self.state_manager, "ai_agent_node", None)
        if ai:
            ai.set_context(name=disp)

        # Ask next unknown's name if queue not empty
        if remaining:
            msg = f"Nice to meet you, {disp}! And what's your name?"
            self._speak(msg)
            self._chat(msg)
            print(f"[FR] Name gate → next: {remaining[0]}")
        else:
            print("[FR] Name gate — all unknowns identified")

        return True

    def set_visitor_preference(self, preference: str, vid: str = None) -> bool:
        pl       = preference.lower().strip()
        detected = next(
            (t for t, kws in self._PREFERENCE_KEYWORDS.items()
             if any(k in pl for k in kws)),
            pl if pl in self._FUN_FACTS else None,
        )
        if not detected:
            return False
        target = None
        with self._lock:
            if vid and vid in self.visitors:
                self.visitors[vid]["preference"] = detected
                target = vid
            else:
                for v, d in self.visitors.items():
                    if (not d.get("name", "").startswith("Visitor_")
                            and d.get("visits", 0) == 1
                            and not d.get("preference")):
                        d["preference"] = detected
                        target = v
                        break
        if target:
            self._write_db()
            print(f"[FR] Preference '{detected}' → {target}")
            return True
        return False

    # ── Fun facts ─────────────────────────────────────────────────────────────

    _FUN_FACTS = {
        "science": [
            "Fun fact: a teaspoon of neutron star would weigh about 10 million tons!",
            "Did you know? Honey never expires — archaeologists found 3000-year-old honey in Egyptian tombs!",
            "Fun fact: Sharks are older than trees — they've existed for over 400 million years!",
        ],
        "sport": [
            "Fun fact: the first Olympic games in 776 BC had only one event — a foot race!",
            "Did you know? Usain Bolt's top speed was 44.72 km/h — faster than a horse!",
        ],
        "history": [
            "Fun fact: Cleopatra lived closer in time to the Moon landing than to the Great Pyramid!",
            "Did you know? Oxford University is older than the Aztec Empire!",
            "Fun fact: the shortest war in history lasted only 38 minutes!",
        ],
        "tech": [
            "Fun fact: the first computer bug was a real moth found in a Harvard computer in 1947!",
            "Did you know? Google was almost named Backrub!",
            "Did you know? Email is older than the internet — invented in 1971!",
        ],
        "art": [
            "Fun fact: Van Gogh sold only one painting in his lifetime!",
            "Did you know? Beethoven was completely deaf when he composed his Ninth Symphony!",
            "Fun fact: the Eiffel Tower grows 15 cm taller every summer!",
        ],
    }

    _TIME_FACTS = {
        "morning":   [
            "Fun fact: the brain is most creative in the first two hours after waking up!",
            "Did you know? Coffee was discovered by a goat herder in Ethiopia around 850 AD!",
        ],
        "midday":    [
            "Fun fact: the most productive time of day is between 10 AM and noon!",
            "Did you know? Smiling — even briefly — reduces stress hormones!",
        ],
        "afternoon": [
            "Fun fact: mild fatigue actually boosts creativity!",
            "Did you know? Your body temperature naturally dips at 2 PM!",
        ],
        "evening":   [
            "Fun fact: people make better long-term decisions in the evening!",
            "Did you know? The brain consolidates memories during sleep!",
        ],
    }

    _PREFERENCE_KEYWORDS = {
        "science":  ["science","physics","chemistry","biology","nature","space",
                     "علوم","فيزياء","كيمياء","بيولوجيا","طبيعة","فضاء"],
        "sport":    ["sport","sports","football","basketball","tennis","soccer","gym",
                     "رياضة","كرة","رياضي","تنس","سباحة","ألعاب"],
        "history":  ["history","historical","ancient","war","civilization",
                     "تاريخ","تاريخي","قديم","حرب","حضارة"],
        "tech":     ["tech","technology","computer","coding","programming","ai","software",
                     "تكنولوجيا","تقنية","كمبيوتر","برمجة","ذكاء","تطوير"],
        "art":      ["art","music","painting","drawing","culture","cinema","literature",
                     "فن","موسيقى","رسم","ثقافة","سينما","أدب","فنون"],
    }

    def _get_fun_fact(self, vid=None, visits=1) -> str:
        pref = None
        if vid:
            with self._lock:
                pref = self.visitors.get(vid, {}).get("preference")
        if pref and pref in self._FUN_FACTS:
            return random.choice(self._FUN_FACTS[pref])
        h   = time.localtime().tm_hour
        key = ("morning" if h < 10 else "midday" if h < 13
               else "afternoon" if h < 17 else "evening")
        return random.choice(self._TIME_FACTS[key])

    # ── Geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(b1, b2) -> float:
        if not b1 or not b2:
            return 0.0
        x1,y1,x2,y2 = b1
        x3,y3,x4,y4 = b2
        ix1,iy1 = max(x1,x3), max(y1,y3)
        ix2,iy2 = min(x2,x4), min(y2,y4)
        inter   = max(0, ix2-ix1) * max(0, iy2-iy1)
        union   = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / union if union > 0 else 0.0

    # ── InsightFace detect + encode (primary) ─────────────────────────────────

    def _detect_encode_insight(self, frame) -> list:
        """RetinaFace detection + ArcFace 512-dim encoding in one pass."""
        if self.insight_app is None:
            return []
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.insight_app.get(rgb)
            if not faces:
                return []
            h_f, w_f = frame.shape[:2]
            result = []
            for face in faces:
                if face.det_score < 0.5:
                    continue
                b   = face.bbox.astype(int)
                box = (max(0,b[0]), max(0,b[1]),
                       min(w_f,b[2]), min(h_f,b[3]))
                emb = face.normed_embedding
                result.append((box, emb))
            if len(result) > 1:
                print(f"[FR] {len(result)} faces detected simultaneously")
            return result
        except Exception as e:
            print(f"[FR] InsightFace error: {e}")
            return []

    # ── YOLO + dlib (fallback) ────────────────────────────────────────────────

    def _detect_yolo(self, frame) -> list:
        if not YOLO_OK or self.yolo is None:
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
        age = self.frame_ctr - self._last_box_frame
        if self._last_boxes and age < self._BOX_CACHE_FRAMES:
            return self._last_boxes
        try:
            h_orig, w_orig = frame.shape[:2]
            scale = self._DETECT_SIZE / max(h_orig, w_orig)
            if scale < 1.0:
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LINEAR)
            else:
                small, scale = frame, 1.0
            res   = self.yolo(small, conf=0.45, verbose=False)[0]
            boxes = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append((int(x1/scale), int(y1/scale),
                               int(x2/scale), int(y2/scale)))
            self._last_boxes     = boxes
            self._last_box_frame = self.frame_ctr
            return boxes
        except Exception as e:
            print(f"[FR] YOLO: {e}")
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]

    def _encode_dlib(self, enh_frame, box):
        if not DLIB_OK or not CV2_OK:
            return None
        try:
            x1, y1, x2, y2 = box
            h, w = enh_frame.shape[:2]
            m    = 10
            crop = enh_frame[max(0,y1-m):min(h,y2+m),
                             max(0,x1-m):min(w,x2+m)]
            if crop.size == 0:
                return None
            rgb      = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ch, cw   = rgb.shape[:2]
            face_loc = (m, cw-m, ch-m, m)
            encs = face_recognition.face_encodings(
                rgb, [face_loc], num_jitters=self.jitters)
            if encs:
                return encs[0]
            locs = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                return None
            encs = face_recognition.face_encodings(
                rgb, locs, num_jitters=self.jitters)
            return encs[0] if encs else None
        except Exception as e:
            print(f"[FR] dlib encode: {e}")
            return None

    # ── Match ─────────────────────────────────────────────────────────────────

    def _match(self, enc):
        with self._lock:
            snap = dict(self._enc_cache)
        if not snap:
            return None, 1.0
        try:
            ids  = list(snap.keys())
            arrs = list(snap.values())
            if self.insight_app is not None:
                sims = [float(np.dot(enc, a)) for a in arrs]
                best = int(np.argmax(sims))
                sim  = sims[best]
                dist = 1.0 - sim
                vid  = ids[best] if sim >= self._INSIGHT_THR else None
            else:
                dists = face_recognition.face_distance(arrs, enc)
                best  = int(dists.argmin())
                dist  = float(dists[best])
                vid   = ids[best] if dist < self.db_thr else None
            with self._lock:
                name = self.visitors.get(ids[best], {}).get("name", "?")
            print(f"[FR] {'✓' if vid else '✗'} {name} dist={dist:.3f}")
            return vid, dist
        except Exception as e:
            print(f"[FR] match: {e}")
            return None, 1.0

    def _update_encoding(self, vid, new_enc):
        with self._lock:
            if vid not in self.visitors or "encoding" not in self.visitors[vid]:
                return
            old     = self._enc_cache.get(
                vid, np.asarray(self.visitors[vid]["encoding"],
                                dtype=np.float64))
            updated = 0.7 * old + 0.3 * np.asarray(new_enc, dtype=np.float64)
            if self.insight_app is not None:
                norm = np.linalg.norm(updated)
                if norm > 0:
                    updated = updated / norm
            self.visitors[vid]["encoding"] = updated.tolist()
            self._enc_cache[vid]           = updated
        self._mark_dirty()

    # ── Track management ──────────────────────────────────────────────────────

    def _get_track(self, enc, box) -> str:
        best_tid, best_score = None, -1.0
        for tid, track in self.tracks.items():
            if track.last_box is None:
                continue
            dist    = track.dist_to(enc)
            iou_val = self._iou(track.last_box, box)
            score   = iou_val * 0.65 + (1.0 - min(dist, 1.0)) * 0.35
            if score > best_score:
                best_score, best_tid = score, tid
        if best_tid and best_score > 0.2:
            return best_tid
        self._tid_ctr += 1
        tid = f"T{self._tid_ctr:03d}"
        self.tracks[tid] = Track(tid)
        print(f"[FR] New track {tid}")
        return tid

    # ── Async worker ──────────────────────────────────────────────────────────

    def _worker(self):
        while self.running:
            try:
                enh = self._rec_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.insight_app is not None:
                    face_pairs = self._detect_encode_insight(enh)
                else:
                    face_pairs = []
                    for box in self._detect_yolo(enh):
                        enc = self._encode_dlib(enh, box)
                        if enc is not None:
                            face_pairs.append((box, enc))
                if face_pairs:
                    self._handle_results(face_pairs)
            except Exception as e:
                import traceback
                print(f"[FR] worker: {e}")
                traceback.print_exc()

    def _handle_results(self, face_pairs: list):
        active_tids   = []
        confirmed_map = {}

        for box, enc in face_pairs:
            if enc is None:
                continue
            tid   = self._get_track(enc, box)
            track = self.tracks[tid]
            track.touch(enc, box)
            active_tids.append(tid)

            vid, dist = self._match(enc)
            track.vote(vid, dist)

            update_thr = 0.25 if self.insight_app else 0.28
            if vid and dist < update_thr:
                self._update_encoding(vid, enc)

            conf = track.confirmed_vid()
            if conf:
                confirmed_map[tid] = conf
                self._vid2tid[conf] = tid
            elif track.unknown_confirmed() and not track.registered:
                track.registered = True
                new_vid = self._register_unknown(enc)
                self._vid2tid[new_vid] = tid
                if tid not in self._unk_tids:
                    self._unk_tids.append(tid)
                if self.current_visitor_id is None:
                    self.current_visitor_id = new_vid

        to_greet = []
        for tid in active_tids:
            track = self.tracks.get(tid)
            if track is None or track.greeted:
                continue
            if tid in confirmed_map:
                conf_vid = confirmed_map[tid]
                name     = self._dname(conf_vid)
                if name and name in self._greeted_names:
                    track.greeted = True
                    continue
                to_greet.append((tid, conf_vid))
            elif tid in self._unk_tids:
                to_greet.append((tid, None))
            else:
                stuck = time.time() - track.first_seen
                if stuck > 5.0 and not track.registered and track.last_enc is not None:
                    print(f"[FR] Track {tid} stuck {stuck:.1f}s — forcing unknown")
                    track.registered = True
                    nv = self._register_unknown(track.last_enc)
                    self._vid2tid[nv] = tid
                    self._unk_tids.append(tid)
                    if self.current_visitor_id is None:
                        self.current_visitor_id = nv
                    to_greet.append((tid, None))

        if to_greet:
            self._greet(to_greet, confirmed_map)
            for tid, _ in to_greet:
                if tid in self.tracks:
                    self.tracks[tid].greeted = True

        self._update_overlay(active_tids, confirmed_map)

    # ── Frame entry ───────────────────────────────────────────────────────────

    def process_frame(self, frame):
        if not CV2_OK:
            return
        self.frame_ctr += 1
        if self.frame_ctr % 4 != 0:
            return
        if time.time() - self.last_rec < self.rec_interval:
            return
        self.last_rec = time.time()
        enh = _clahe(frame)
        try:
            self._rec_q.put_nowait(enh)
        except queue.Full:
            pass
        self._flush()

    # ── Registration ──────────────────────────────────────────────────────────

    def _register_unknown(self, enc) -> str:
        vid = f"Visitor_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.visitors[vid] = {
                "encoding":   enc.tolist() if hasattr(enc, "tolist") else list(enc),
                "name":       vid,
                "visits":     1,
                "first_seen": time.strftime("%Y-%m-%d %H:%M"),
                "last_seen":  time.strftime("%Y-%m-%d %H:%M"),
            }
        print(f"[FR] Unknown registered: {vid}")
        return vid

    def _increment_visit(self, vid):
        with self._lock:
            if vid not in self.visitors:
                return
            v = self.visitors[vid]
            v["visits"]    = v.get("visits", 0) + 1
            v["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
        self._mark_dirty()

    # ── Greeting ──────────────────────────────────────────────────────────────

    def _enqueue_unknowns(self, vids: list):
        with self._name_lock:
            for v in vids:
                if v not in self._name_queue:
                    self._name_queue.append(v)
                    print(f"[FR] Name queue ← {v} "
                          f"(total: {len(self._name_queue)})")

    def _greet(self, to_greet, confirmed_map):
        known_names, known_visits, known_vids = [], {}, {}
        unk_vids_to_queue: list[str] = []
        incremented: set = set()

        for tid, conf_vid in to_greet:
            if conf_vid:
                name = self._dname(conf_vid)
                if name and name not in known_names:
                    known_names.append(name)
                    known_vids[name] = conf_vid
                    if conf_vid not in incremented:
                        self._increment_visit(conf_vid)
                        incremented.add(conf_vid)
                    with self._lock:
                        known_visits[name] = (
                            self.visitors.get(conf_vid, {}).get("visits", 1))
            else:
                for v, t in self._vid2tid.items():
                    if t == tid and tid in self._unk_tids:
                        unk_vids_to_queue.append(v)
                        break

        self._enqueue_unknowns(unk_vids_to_queue)
        unk_count = len(unk_vids_to_queue)
        msg = ""

        if known_names and not unk_count:
            if len(known_names) == 1:
                n   = known_names[0]
                v   = known_visits.get(n, 1)
                vid = known_vids.get(n)
                if   v == 1: g = f"Hello {n}, welcome! Great to meet you."
                elif v == 2: g = f"Welcome back {n}! Great to see you again."
                elif v == 3: g = f"Hello again {n}! Your third visit already."
                else:        g = f"Welcome back {n}!"
                msg = f"{g} {self._get_fun_fact(vid=vid, visits=v)}"
            else:
                ns  = ", ".join(known_names[:-1]) + f" and {known_names[-1]}"
                msg = f"Hello {ns}, welcome back! {self._get_fun_fact()}"
            for n in known_names:
                self._greeted_names.add(n)
            self._speak_greeting(msg)

        elif not known_names:
            if unk_count == 1:
                msg = "Hello! I don't think we've met before. What's your name?"
            else:
                msg = (f"Hello everyone! I see {unk_count} new faces. "
                       f"Let's get acquainted — what's your name?")
            self._speak_greeting(msg)

        else:
            ks = (known_names[0] if len(known_names) == 1
                  else ", ".join(known_names[:-1]) + f" and {known_names[-1]}")
            msg = (f"Welcome back {ks}! "
                   + ("And I see someone new — what's your name?"
                      if unk_count == 1
                      else f"And {unk_count} new faces — what's your name?"))
            for n in known_names:
                self._greeted_names.add(n)
            self._speak_greeting(msg)

        if msg:
            self._chat(msg)
        self.state_manager.change_state("GREETING")
        self.state_manager.command_queue.put({"type": "VISITOR_ARRIVED"})
        print(f"[FR] Greeting: {msg}")

    def _speak_greeting(self, text):
        with self._speech_lock:
            now = time.time()
            if now - self._last_greeting_speech < self._speech_gap:
                return
            self._last_greeting_speech = now
        self._speak(text)

    def _speak_goodbye(self, text):
        with self._speech_lock:
            now = time.time()
            if now - self._last_goodbye_speech < self._speech_gap:
                return
            self._last_goodbye_speech = now
        self._speak(text)

    # ── Overlay ───────────────────────────────────────────────────────────────

    def _update_overlay(self, active_tids, confirmed_map):
        sensor = (getattr(self.main_system, "sensor_node", None)
                  if self.main_system else None)
        if not sensor or not hasattr(sensor, "set_face_overlays"):
            return
        overlays = []
        for tid in active_tids:
            track = self.tracks.get(tid)
            if track is None or track.last_box is None:
                continue
            conf = confirmed_map.get(tid)
            if conf:
                name        = self._dname(conf) or "?"
                label, known = name, name != "?"
            elif tid in self._unk_tids:
                with self._name_lock:
                    pos = next(
                        (i+1 for i, v in enumerate(self._name_queue)
                         if self._vid2tid.get(v) == tid), None)
                label = f"? Name? (#{pos})" if pos else "? Who are you?"
                known = False
            else:
                label, known = "Recognizing...", False
            overlays.append({"name": label, "known": known,
                             "bbox": track.last_box})
        sensor.set_face_overlays(overlays)

        gui = getattr(self.state_manager, "gui_node", None)
        if gui and overlays:
            known_names = [o["name"] for o in overlays if o["known"]]
            unk_cnt     = sum(1 for o in overlays if not o["known"])
            if known_names:
                display = ", ".join(known_names)
                if unk_cnt:
                    display += f" +{unk_cnt}?"
                gui.update_visitor_info(name=display)

    # ── Absence ───────────────────────────────────────────────────────────────

    def _check_absences(self):
        departed = []
        for tid, track in list(self.tracks.items()):
            if track.absent(self.absence_thr) and not track.goodbye_sent:
                track.goodbye_sent = True
                departed.append(tid)

        for tid in departed:
            track = self.tracks[tid]
            conf  = track.confirmed_vid()
            dn    = self._dname(conf) if conf else None
            if dn:
                msg = f"Goodbye, {dn}! Have a wonderful day."
                self._speak_goodbye(msg)
                self._chat(msg)
                print(f"[FR] Goodbye: {dn}")
                self._greeted_names.discard(dn)

            for v, t in list(self._vid2tid.items()):
                if t != tid:
                    continue
                with self._name_lock:
                    if v in self._name_queue:
                        self._name_queue.remove(v)
                        print(f"[FR] Name queue removed {v} (visitor left)")
                if self.current_visitor_id == v:
                    self.current_visitor_id = None
                dn2 = self._dname(v)
                if dn2:
                    self._greeted_names.discard(dn2)
                del self._vid2tid[v]
                with self._lock:
                    if self.visitors.get(v, {}).get(
                            "name", "").startswith("Visitor_"):
                        self.visitors.pop(v, None)
                        self._enc_cache.pop(v, None)

            if tid in self._unk_tids:
                self._unk_tids.remove(tid)
            del self.tracks[tid]

        if self.current_visitor_id not in self._vid2tid:
            self.current_visitor_id = None

        if departed and not self.tracks:
            with self._name_lock:
                self._name_queue.clear()
            self.current_visitor_id = None
            self.state_manager.change_state("IDLE")
            self.state_manager.command_queue.put({"type": "VISITOR_LEFT"})
            s = (getattr(self.main_system, "sensor_node", None)
                 if self.main_system else None)
            if s and hasattr(s, "set_face_overlays"):
                s.set_face_overlays([])
            ai = getattr(self.state_manager, "ai_agent_node", None)
            if ai and hasattr(ai, "history"):
                ai.set_context(name=None, visits=0)
                ai.history = []
            gui = getattr(self.state_manager, "gui_node", None)
            if gui:
                gui.update_visitor_info(name="—", visits="—", mood="—")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _speak(self, text):
        try:
            self.sound_q.put({"type": "SPEAK", "text": text})
        except Exception:
            pass

    def _chat(self, text):
        gui = getattr(self.state_manager, "gui_node", None)
        if gui:
            gui.add_chat_message("ROBOT", text)

    # ── Run / stop ────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        engine = ("InsightFace ArcFace" if self.insight_app
                  else f"dlib jitters={self.jitters}")
        print(f"[FR] Running ({engine} | vote {Track.VOTE_NEEDED}/{Track.VOTE_SIZE} "
              f"| async worker | multi-face name queue)…")

        threading.Thread(target=self._worker,
                         daemon=True, name="FR-worker").start()

        def _absence_loop():
            while self.running:
                time.sleep(1.0)
                try:
                    self._check_absences()
                except Exception as e:
                    print(f"[FR] absence: {e}")

        threading.Thread(target=_absence_loop,
                         daemon=True, name="FR-absence").start()

        while self.running:
            try:
                if not self.frame_queue.empty():
                    self.process_frame(self.frame_queue.get_nowait())
            except Exception:
                pass
            time.sleep(0.02)

        if self._dirty:
            self._write_db()

    def stop(self):
        self.running = False
        if self._dirty:
            self._write_db()
        print("[FR] Stopped")