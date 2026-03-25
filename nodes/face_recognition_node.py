"""
Face Recognition Node — Complete Fixed Version
=================================================================
FIXES APPLIED:
1. FIX 5 — set_visitor_name() NO LONGER puts ASK_PREFERENCE on the
   command queue. state_manager's SET_NAME handler is the single source
   of truth for the greeting + preference flow. Removing it here prevents
   the double preference question that was occurring before.

2. _greet() msg variable is always initialized to '' so _chat(msg) can
   never raise UnboundLocalError.

3. _get_fun_fact() is a single English path with no language detection.

4. All greeting/goodbye messages are English only.
"""

import time, os, pickle, uuid, threading, random
from collections import Counter

try:
    import cv2, numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

try:
    import face_recognition
    DLIB_OK = True
except ImportError:
    DLIB_OK = False
    print("[FR] pip install face-recognition")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clahe(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l = float(l.mean())
        std_l  = float(l.std())
        clip_limit = max(2.0, min(5.0, 3.0 + (30 - std_l) / 10))
        eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
        if mean_l < 80:
            gamma = 0.5 if mean_l < 50 else 0.7
            table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype('uint8')
            eq = cv2.LUT(eq, table)
        elif mean_l > 180:
            gamma = 1.8 if mean_l > 200 else 1.4
            table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype('uint8')
            eq = cv2.LUT(eq, table)
        return cv2.cvtColor(cv2.merge([eq, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


# ── Track ─────────────────────────────────────────────────────────────────────

class Track:
    VOTE_SIZE   = 6
    VOTE_NEEDED = 3

    def __init__(self, tid):
        self.tid          = tid
        self.first_seen   = time.time()
        self.last_seen    = time.time()
        self.last_enc     = None
        self.last_box     = None
        self.votes        = []
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

    def unknown_confirmed(self):
        if len(self.votes) < self.VOTE_NEEDED:
            return False
        return sum(1 for v, _ in self.votes if v is None) >= self.VOTE_NEEDED

    def absent(self, thr):
        return time.time() - self.last_seen > thr

    def dist_to(self, enc):
        if self.last_enc is None or not DLIB_OK:
            return 1.0
        try:
            return float(face_recognition.face_distance([self.last_enc], enc)[0])
        except Exception:
            return 1.0


# ── Node ──────────────────────────────────────────────────────────────────────

class FaceRecognitionNode:
    def __init__(self, frame_queue, sound_emotion_queue, state_manager,
                 main_system=None,
                 db_path='visitor_database.pkl',
                 yolo_model='yolov8n.pt',
                 db_thr=0.42,
                 track_thr=0.55,
                 absence_thr=10.0,
                 rec_interval=0.8,
                 jitters=5):
        self.frame_queue   = frame_queue
        self.sound_q       = sound_emotion_queue
        self.state_manager = state_manager
        self.main_system   = main_system
        self.running       = False
        self.db_path       = db_path
        self.db_thr        = db_thr
        self.track_thr     = track_thr
        self.absence_thr   = absence_thr
        self.rec_interval  = rec_interval
        self.jitters       = jitters
        self.visitors      = self._load_db()
        self._lock         = threading.Lock()

        self.yolo = None
        if YOLO_OK:
            try:
                self.yolo = YOLO(yolo_model)
                print(f"[FR] YOLO: {yolo_model}")
            except Exception as e:
                print(f"[FR] YOLO error: {e}")

        self.tracks: dict[str, Track] = {}
        self._tid_ctr  = 0
        self._vid2tid  = {}
        self._unk_tids = []

        self.current_visitor_id      = None
        self.current_visitor_name    = None
        self.pending_name_visitor_id = None

        self.last_rec       = 0.0
        self.frame_ctr      = 0
        self._greeted_names = set()
        self._speech_lock   = threading.Lock()

        self._last_goodbye_speech  = 0.0
        self._last_greeting_speech = 0.0
        self._speech_gap           = 3.0

        print(f"[FR] Ready — {len(self.visitors)} visitors | "
              f"dlib jitters={jitters} db_thr={db_thr}")

    # ── DB ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_valid_name(name):
        if not name or len(name) < 2:
            return False
        if name.startswith('Visitor_'):
            return False
        if len(name.split()) > 3:
            return False
        if any(c in name for c in ['?', '!', '\\', '/']):
            return False
        if name.strip().endswith(','):
            return False
        NOT_NAMES = {
            'hi', 'it', 'ess', 'monnaie', 'done', 'think', 'met', 'before',
            'what', 'your', 'name', 'backslash', 'typically', 'definkly',
            'you', 'we', 'the', 'and', 'yes', 'no', 'ok', 'okay', 'back',
        }
        if name.strip().lower() in NOT_NAMES:
            return False
        alpha = sum(1 for c in name if c.isalpha() or '\u0600' <= c <= '\u06FF')
        if alpha < 2:
            return False
        arabic_chars = sum(1 for c in name if '\u0600' <= c <= '\u06FF')
        latin_chars  = sum(1 for c in name if c.isalpha() and ord(c) < 0x600)
        if arabic_chars > 0 and latin_chars == 0 and len(name.strip()) < 3:
            return False
        return True

    def _load_db(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    db = pickle.load(f)
                no_placeholders = {k: v for k, v in db.items()
                                   if not v.get('name', '').startswith('Visitor_')}
                clean = {k: v for k, v in no_placeholders.items()
                         if self._is_valid_name(v.get('name', ''))}
                removed = len(db) - len(clean)
                if removed:
                    print(f"[FR] Cleaned {removed} invalid/unnamed entries from DB")
                print(f"[FR] DB: {[v['name'] for v in clean.values()]}")
                return clean
        except Exception as e:
            print(f"[FR] DB load: {e}")
        return {}

    def _save_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
            named = {k: v for k, v in self.visitors.items()
                     if not v.get('name', '').startswith('Visitor_')}
            with open(self.db_path, 'wb') as f:
                pickle.dump(named, f)
        except Exception as e:
            print(f"[FR] DB save: {e}")

    # ── Public gate API ───────────────────────────────────────────────────────

    def is_waiting_for_name(self):
        vid = self.pending_name_visitor_id
        if vid is None:
            return False
        entry = self.visitors.get(vid)
        if entry is None:
            self.pending_name_visitor_id = None
            return False
        name = entry.get('name', '')
        if not name.startswith('Visitor_'):
            self.pending_name_visitor_id = None
            return False
        return True

    def set_visitor_name(self, visitor_id_or_name, name=None):
        if name is None:
            actual_name = visitor_id_or_name
        else:
            actual_name = name

        actual_name = actual_name.strip()
        # Preserve Arabic casing; titlecase only Latin names
        arabic_chars = sum(1 for c in actual_name if '\u0600' <= c <= '\u06FF')
        latin_chars  = sum(1 for c in actual_name if c.isalpha() and ord(c) < 0x600)
        if latin_chars > arabic_chars:
            actual_name = actual_name.title()

        if not actual_name or len(actual_name) < 2:
            return False

        vid = self.pending_name_visitor_id
        if vid is None:
            vid = self.current_visitor_id
        if vid is None:
            print(f"[FR] set_visitor_name('{actual_name}'): no pending visitor — ignored")
            return False

        if vid not in self.visitors:
            print(f"[FR] set_visitor_name: {vid} not in DB — ignored")
            self.pending_name_visitor_id = None
            return False

        self.visitors[vid]['name'] = actual_name
        self._save_db()
        print(f"[FR] ✓ Name saved: '{actual_name}' → {vid}")

        tid = self._vid2tid.get(vid)
        if tid and tid in self.tracks:
            self.tracks[tid].greeted = True

        if vid in self._unk_tids:
            self._unk_tids.remove(vid)
        track_id = self._vid2tid.get(vid)
        if track_id and track_id in self._unk_tids:
            self._unk_tids.remove(track_id)

        self.pending_name_visitor_id = None
        self.current_visitor_id      = None
        self.current_visitor_name    = actual_name

        ai = getattr(self.state_manager, 'ai_agent_node', None)
        if ai:
            ai.set_context(name=actual_name)

        # FIX 5 — Do NOT put ASK_PREFERENCE here.
        # state_manager's SET_NAME handler handles preference in one unified path.
        # Putting ASK_PREFERENCE here as well caused a double preference question.

        return True

    # ── Fun facts (English only) ──────────────────────────────────────────────

    _FUN_FACTS = {
        'science': [
            "Fun fact: a teaspoon of neutron star would weigh about 10 million tons!",
            "Did you know? Honey never expires — archaeologists found 3000-year-old honey in Egyptian tombs!",
            "Fun fact: your body contains enough DNA to stretch from Earth to Pluto and back 17 times!",
            "Did you know? Sharks are older than trees — they've existed for over 400 million years!",
            "Fun fact: a day on Venus is longer than a year on Venus!",
            "Did you know? Water can boil and freeze at the same time — it's called the triple point!",
        ],
        'sport': [
            "Fun fact: the first Olympic games in 776 BC had only one event — a foot race!",
            "Did you know? A golf ball has exactly 336 dimples, each precision-engineered for lift!",
            "Fun fact: Usain Bolt's top speed was 44.72 km/h — faster than a horse over short distances!",
            "Did you know? Basketball was invented with a peach basket — you had to climb to retrieve the ball!",
            "Fun fact: the Tour de France burns about 123,000 calories per rider!",
            "Did you know? Table tennis balls can reach over 150 km/h in professional matches!",
        ],
        'history': [
            "Fun fact: Cleopatra lived closer in time to the Moon landing than to the Great Pyramid!",
            "Did you know? Oxford University is older than the Aztec Empire!",
            "Fun fact: the shortest war in history lasted only 38 minutes — Britain vs Zanzibar in 1896!",
            "Did you know? Vikings gave kittens to new brides as essential household gifts!",
            "Fun fact: Napoleon was once attacked by a swarm of rabbits during a hunt he organized!",
            "Did you know? Ancient Egyptians used moldy bread as an antibiotic centuries before penicillin!",
        ],
        'tech': [
            "Fun fact: the first computer bug was a real moth found inside a Harvard computer in 1947!",
            "Did you know? Google was almost named Backrub!",
            "Fun fact: there are more possible chess games than atoms in the observable universe!",
            "Did you know? Email is older than the internet — it was invented in 1971!",
            "Fun fact: 90% of the world's data was created in just the last two years!",
            "Did you know? The original WiFi name was IEEE 802.11b Direct Sequence!",
        ],
        'art': [
            "Fun fact: Van Gogh sold only one painting in his lifetime — now they sell for over 80 million dollars!",
            "Did you know? The Mona Lisa has no eyebrows — it was fashionable in Renaissance Florence!",
            "Fun fact: Picasso's first word was 'piz' — short for lápiz, meaning pencil!",
            "Did you know? Beethoven was completely deaf when he composed his Ninth Symphony!",
            "Fun fact: the Eiffel Tower grows 15 cm taller every summer due to thermal expansion!",
            "Did you know? The shortest poem ever written is by Ogden Nash — just three words!",
        ],
    }

    _TIME_FACTS = {
        'morning':   [
            "Fun fact: the brain is most creative in the first two hours after waking up!",
            "Did you know? Coffee was discovered by a goat herder in Ethiopia around 850 AD!",
        ],
        'midday':    [
            "Fun fact: the most productive time of day is between 10 AM and noon!",
            "Did you know? Smiling — even briefly — reduces stress hormones!",
        ],
        'afternoon': [
            "Fun fact: mild fatigue actually boosts creativity — you might be at peak inspiration right now!",
            "Did you know? Your body temperature naturally dips at 2 PM!",
        ],
        'evening':   [
            "Fun fact: people make better long-term decisions in the evening!",
            "Did you know? The brain consolidates memories during sleep — tonight matters!",
        ],
    }

    _PREFERENCE_KEYWORDS = {
        'science':  ['science', 'physics', 'chemistry', 'biology', 'nature', 'space',
                     'علوم', 'فيزياء', 'كيمياء', 'بيولوجيا', 'طبيعة', 'فضاء'],
        'sport':    ['sport', 'sports', 'football', 'basketball', 'tennis', 'soccer', 'gym',
                     'رياضة', 'كرة', 'رياضي', 'تنس', 'سباحة', 'ألعاب'],
        'history':  ['history', 'historical', 'ancient', 'war', 'civilization',
                     'تاريخ', 'تاريخي', 'قديم', 'حرب', 'حضارة'],
        'tech':     ['tech', 'technology', 'computer', 'coding', 'programming', 'ai', 'software',
                     'تكنولوجيا', 'تقنية', 'كمبيوتر', 'برمجة', 'ذكاء', 'تطوير'],
        'art':      ['art', 'music', 'painting', 'drawing', 'culture', 'cinema', 'literature',
                     'فن', 'موسيقى', 'رسم', 'ثقافة', 'سينما', 'أدب', 'فنون'],
    }

    def _get_fun_fact(self, vid=None, visits=1):
        """Returns a personalized fun fact in English based on saved preference."""
        preference = None
        if vid and vid in self.visitors:
            preference = self.visitors[vid].get('preference')

        if preference and preference in self._FUN_FACTS:
            return random.choice(self._FUN_FACTS[preference])

        hour = time.localtime().tm_hour
        if hour < 10:
            return random.choice(self._TIME_FACTS['morning'])
        elif hour < 13:
            return random.choice(self._TIME_FACTS['midday'])
        elif hour < 17:
            return random.choice(self._TIME_FACTS['afternoon'])
        else:
            return random.choice(self._TIME_FACTS['evening'])

    def set_visitor_preference(self, preference: str, vid: str = None) -> bool:
        pref_low = preference.lower().strip()
        detected = None
        for topic, keywords in self._PREFERENCE_KEYWORDS.items():
            if any(k in pref_low for k in keywords):
                detected = topic
                break
        if not detected and pref_low in self._FUN_FACTS:
            detected = pref_low
        if not detected:
            print(f"[FR] Could not detect preference from: '{preference}'")
            return False

        if vid and vid in self.visitors:
            self.visitors[vid]['preference'] = detected
            self._save_db()
            print(f"[FR] ✓ Preference saved: '{detected}' → {vid}")
            return True

        for v, data in self.visitors.items():
            name = data.get('name', '')
            if (not name.startswith('Visitor_') and
                    data.get('visits', 0) == 1 and
                    not data.get('preference')):
                self.visitors[v]['preference'] = detected
                self._save_db()
                print(f"[FR] ✓ Preference saved: '{detected}' → {v} ({name})")
                return True

        print(f"[FR] set_visitor_preference: no matching visitor found")
        return False

    # ── IOU / encoding helpers ────────────────────────────────────────────────

    def _iou(self, box1, box2):
        if not box1 or not box2:
            return 0.0
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        ix1, iy1 = max(x1, x3), max(y1, y3)
        ix2, iy2 = min(x2, x4), min(y2, y4)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
        return inter / union if union > 0 else 0.0

    def _update_encoding(self, vid, new_enc):
        with self._lock:
            if vid not in self.visitors or 'encoding' not in self.visitors[vid]:
                return
            old     = np.array(self.visitors[vid]['encoding'])
            updated = 0.7 * old + 0.3 * np.array(new_enc)
            self.visitors[vid]['encoding'] = updated.tolist()
        self._save_db()

    def _detect(self, frame):
        if not YOLO_OK or self.yolo is None:
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]
        with self._lock:
            try:
                res = self.yolo(frame, classes=[0], conf=0.35, verbose=False)[0]
                return [tuple(map(int, b.xyxy[0].tolist())) for b in res.boxes]
            except Exception as e:
                print(f"[FR] YOLO: {e}")
                h, w = frame.shape[:2]
                return [(0, 0, w, h)]

    def _encode(self, frame, box):
        if not DLIB_OK or not CV2_OK:
            return None
        with self._lock:
            try:
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                m    = 30
                crop = frame[max(0,y1-m):min(h,y2+m), max(0,x1-m):min(w,x2+m)]
                if crop.size == 0:
                    return None
                enhanced = clahe(crop)
                small    = cv2.resize(enhanced, (0,0), fx=0.5, fy=0.5)
                rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs     = face_recognition.face_locations(rgb, model='hog')
                if not locs:
                    rgb2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb2, model='hog')
                    if not locs:
                        return None
                    encs = face_recognition.face_encodings(rgb2, locs, num_jitters=self.jitters)
                else:
                    encs = face_recognition.face_encodings(rgb, locs, num_jitters=self.jitters)
                return encs[0] if encs else None
            except Exception as e:
                print(f"[FR] encode: {e}")
                return None

    def _match(self, enc):
        if not DLIB_OK:
            return None, 1.0
        with self._lock:
            named = {vid: v for vid, v in self.visitors.items()
                     if not v.get('name', '').startswith('Visitor_')
                     and v.get('encoding') is not None}
        if not named:
            return None, 1.0
        try:
            encs  = [np.array(v['encoding']) for v in named.values()]
            ids   = list(named.keys())
            dists = face_recognition.face_distance(encs, enc)
            best  = int(dists.argmin())
            dist  = float(dists[best])
            vid   = ids[best] if dist < self.db_thr else None
            name  = self.visitors.get(ids[best], {}).get('name', '?')
            print(f"[FR] {'✓' if vid else '✗'} {name} dist={dist:.3f}")
            return vid, dist
        except Exception as e:
            print(f"[FR] match: {e}")
            return None, 1.0

    def _get_track(self, enc, box):
        best_tid, best_score, best_iou, best_dist = None, -1.0, 0.0, 1.0
        for tid, track in self.tracks.items():
            if track.last_box is None:
                continue
            dist    = track.dist_to(enc)
            iou_val = self._iou(track.last_box, box)
            score   = iou_val * 0.65 + (1.0 - min(dist, 1.0)) * 0.35
            if score > best_score:
                best_score, best_tid, best_iou, best_dist = score, tid, iou_val, dist
        if best_tid and (best_iou > 0.25 or best_dist < self.track_thr):
            return best_tid
        self._tid_ctr += 1
        tid = f"T{self._tid_ctr:03d}"
        self.tracks[tid] = Track(tid)
        print(f"[FR] New track {tid}")
        return tid

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(self, frame):
        if not CV2_OK:
            return
        self.frame_ctr += 1
        if self.frame_ctr % 4 != 0:
            self._check_absences()
            return
        now = time.time()
        if now - self.last_rec < self.rec_interval:
            self._check_absences()
            return
        self.last_rec = now
        try:
            enh   = clahe(frame)
            boxes = self._detect(enh)
            if not boxes:
                self._check_absences()
                return
            active_tids   = []
            confirmed_map = {}
            for box in boxes:
                enc = self._encode(enh, box)
                if enc is None:
                    continue
                tid   = self._get_track(enc, box)
                track = self.tracks[tid]
                track.touch(enc, box)
                active_tids.append(tid)
                vid, dist = self._match(enc)
                track.vote(vid, dist)
                if vid and dist < 0.28:
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
                    stuck_secs = time.time() - track.first_seen
                    if stuck_secs > 5.0 and not track.registered and track.last_enc is not None:
                        print(f"[FR] Track {tid} stuck {stuck_secs:.1f}s — forcing unknown")
                        track.registered = True
                        new_vid = self._register_unknown(track.last_enc)
                        self._vid2tid[new_vid] = tid
                        self._unk_tids.append(tid)
                        if self.current_visitor_id is None:
                            self.current_visitor_id = new_vid
                        to_greet.append((tid, None))

            if to_greet:
                self._greet(to_greet, confirmed_map)
                for tid, _ in to_greet:
                    if tid in self.tracks:
                        self.tracks[tid].greeted = True

            self._update_overlay(active_tids, confirmed_map)
        except Exception as e:
            import traceback
            print(f"[FR] Error: {e}")
            traceback.print_exc()
        self._check_absences()

    def _register_unknown(self, enc):
        vid = f"Visitor_{uuid.uuid4().hex[:8]}"
        self.visitors[vid] = {
            'encoding':   enc.tolist() if hasattr(enc, 'tolist') else list(enc),
            'name':       vid,
            'visits':     1,
            'first_seen': time.strftime('%Y-%m-%d %H:%M'),
            'last_seen':  time.strftime('%Y-%m-%d %H:%M'),
        }
        print(f"[FR] Unknown registered: {vid}")
        return vid

    def _increment_visit(self, vid):
        if vid not in self.visitors:
            return
        v = self.visitors[vid]
        v['visits']    = v.get('visits', 0) + 1
        v['last_seen'] = time.strftime('%Y-%m-%d %H:%M')
        self._save_db()

    def _dname(self, vid):
        if vid is None:
            return None
        name = self.visitors.get(vid, {}).get('name', vid)
        return None if (not name or name.startswith('Visitor_')) else name

    # ── Greeting ──────────────────────────────────────────────────────────────

    def _greet(self, to_greet, confirmed_map):
        known_names  = []
        known_visits = {}
        known_vids   = {}
        unk_count    = 0
        incremented  = set()

        for tid, conf_vid in to_greet:
            if conf_vid:
                name = self._dname(conf_vid)
                if name and name not in known_names:
                    known_names.append(name)
                    known_vids[name] = conf_vid
                    if conf_vid not in incremented:
                        self._increment_visit(conf_vid)
                        incremented.add(conf_vid)
                    known_visits[name] = self.visitors.get(conf_vid, {}).get('visits', 1)
            else:
                unk_count += 1

        has_unknown = unk_count > 0

        # Always initialize msg so _chat(msg) never raises UnboundLocalError
        msg = ""

        if known_names and not has_unknown:
            if len(known_names) == 1:
                n, v = known_names[0], known_visits.get(known_names[0], 1)
                vid  = known_vids.get(n)
                if   v == 1: greeting = f"Hello {n}, welcome! Great to meet you."
                elif v == 2: greeting = f"Welcome back {n}! Great to see you again."
                elif v == 3: greeting = f"Hello again {n}! Your third visit already."
                else:        greeting = f"Welcome back {n}!"
                fun_fact = self._get_fun_fact(vid=vid, visits=v)
                msg = f"{greeting} {fun_fact}"
            else:
                ns  = ', '.join(known_names[:-1]) + f' and {known_names[-1]}'
                msg = f"Hello {ns}, welcome back! {self._get_fun_fact()}"
            for n in known_names:
                self._greeted_names.add(n)
            self._speak_greeting(msg)

        elif not known_names:
            msg = ("Hello! I don't think we've met before. What's your name?"
                   if unk_count == 1
                   else f"Hello! I see {unk_count} new people. Could you please introduce yourselves?")
            for tid, conf_vid in to_greet:
                if conf_vid is None and tid in self._unk_tids:
                    for v, t in self._vid2tid.items():
                        if t == tid:
                            self.pending_name_visitor_id = v
                            print(f"[FR] Name gate OPEN → {v}")
                            break
                    break
            self._speak_greeting(msg)

        else:
            ks  = (known_names[0] if len(known_names) == 1
                   else ', '.join(known_names[:-1]) + f' and {known_names[-1]}')
            msg = (f"Welcome back {ks}! And I see someone new — could you please introduce yourself?"
                   if unk_count == 1
                   else f"Welcome back {ks}! And {unk_count} new people — could you introduce yourselves?")
            for n in known_names:
                self._greeted_names.add(n)
            for tid, conf_vid in to_greet:
                if conf_vid is None and tid in self._unk_tids:
                    for v, t in self._vid2tid.items():
                        if t == tid:
                            self.pending_name_visitor_id = v
                            print(f"[FR] Name gate OPEN → {v}")
                            break
                    break
            self._speak_greeting(msg)

        if msg:
            self._chat(msg)
        self.state_manager.change_state('GREETING')
        self.state_manager.command_queue.put({'type': 'VISITOR_ARRIVED'})
        print(f"[FR] Greeting: {msg}")

    def _speak_greeting(self, text):
        with self._speech_lock:
            now = time.time()
            if now - self._last_greeting_speech < self._speech_gap:
                print(f"[FR] Greeting suppressed (too soon): {text[:50]}")
                return
            self._last_greeting_speech = now
        self._speak(text)

    def _speak_goodbye(self, text):
        with self._speech_lock:
            now = time.time()
            if now - self._last_goodbye_speech < self._speech_gap:
                print(f"[FR] Goodbye suppressed (too soon): {text[:50]}")
                return
            self._last_goodbye_speech = now
        self._speak(text)

    def _update_overlay(self, active_tids, confirmed_map):
        sensor = (getattr(self.main_system, 'sensor_node', None)
                  if self.main_system else None)
        if not sensor or not hasattr(sensor, 'set_face_overlays'):
            return
        overlays = []
        for tid in active_tids:
            track = self.tracks.get(tid)
            if track is None or track.last_box is None:
                continue
            conf = confirmed_map.get(tid)
            if conf:
                name  = self._dname(conf) or '?'
                label = name
                known = name != '?'
            elif tid in self._unk_tids:
                label, known = '? Who are you?', False
            else:
                label, known = 'Recognizing...', False
            overlays.append({'name': label, 'known': known, 'bbox': track.last_box})
        sensor.set_face_overlays(overlays)

    # ── Absence / goodbye ─────────────────────────────────────────────────────

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
            if dn:
                self._greeted_names.discard(dn)

            for v, t in list(self._vid2tid.items()):
                if t == tid:
                    if self.pending_name_visitor_id == v:
                        self.pending_name_visitor_id = None
                        print(f"[FR] Name gate CLOSED (visitor left without giving name)")
                    if self.current_visitor_id == v:
                        self.current_visitor_id = None
                    departed_name = self._dname(v)
                    if departed_name:
                        self._greeted_names.discard(departed_name)
                    del self._vid2tid[v]
                    if self.visitors.get(v, {}).get('name', '').startswith('Visitor_'):
                        self.visitors.pop(v, None)

            if tid in self._unk_tids:
                self._unk_tids.remove(tid)
            del self.tracks[tid]

        if self.current_visitor_id is not None:
            if self.current_visitor_id not in self._vid2tid:
                self.current_visitor_id = None

        if departed and not self.tracks:
            self.pending_name_visitor_id = None
            self.current_visitor_id      = None
            self.state_manager.change_state('IDLE')
            self.state_manager.command_queue.put({'type': 'VISITOR_LEFT'})
            s = (getattr(self.main_system, 'sensor_node', None)
                 if self.main_system else None)
            if s and hasattr(s, 'set_face_overlays'):
                s.set_face_overlays([])
            ai = getattr(self.state_manager, 'ai_agent_node', None)
            if ai and hasattr(ai, 'history'):
                ai.set_context(name=None, visits=0)
                ai.history = []
            gui = getattr(self.state_manager, 'gui_node', None)
            if gui:
                gui.update_visitor_info(name='—', visits='—', mood='—')

    # ── Internal speak / chat ─────────────────────────────────────────────────

    def _speak(self, text):
        try:
            self.sound_q.put({'type': 'SPEAK', 'text': text})
        except Exception:
            pass

    def _chat(self, text):
        gui = getattr(self.state_manager, 'gui_node', None)
        if gui:
            gui.add_chat_message('ROBOT', text)

    # ── Run / stop ────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        print(f"[FR] Running (dlib jitters={self.jitters} + YOLO "
              f"+ vote {Track.VOTE_NEEDED}/{Track.VOTE_SIZE} + IOU tracking)…")
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self.process_frame(frame)
            except Exception:
                pass
            time.sleep(0.02)

    def stop(self):
        self.running = False
        print("[FR] Stopped")