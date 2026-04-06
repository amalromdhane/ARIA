"""
Voice Node — Guaranteed Name Recognition v2
=============================================
FIX (on top of previous version):
  Added early TTS guard in _process(): if _preference_mode is True and
  the estimated TTS end time hasn't passed yet, ANY incoming text is
  silently discarded — including Google's partial 'Hist' result that
  arrives while the preference question is still being spoken.
  This prevents the truncated answer from reaching the command queue
  before the full answer ('history') is even recorded.
"""

import threading
import time
import numpy as np

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[VOICE_NODE] pip3 install SpeechRecognition")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[VOICE_NODE] pip3 install pyaudio")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[VOICE_NODE] pip3 install faster-whisper")

try:
    import torch
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
except ImportError:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"


# ─────────────────────────────────────────────────────────────────────────────
# Soundex
# ─────────────────────────────────────────────────────────────────────────────
def _soundex(name: str) -> str:
    name = name.upper().strip()
    if not name:
        return "0000"
    first  = name[0]
    table  = str.maketrans("BFPVCGJKQSXZDTLMNR", "111122222222334556")
    coded  = name[1:].translate(table)
    result = first
    prev   = ''
    for c in coded:
        if c != '0' and c != prev:
            result += c
        prev = c if c != '0' else prev
    return (result + "000")[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Phonetic rescue functions
# ─────────────────────────────────────────────────────────────────────────────
def _prefix_match(heard: str, known_names: list, min_prefix: int = 3):
    h = heard.strip().lower()
    if len(h) < 2:
        return None
    effective_min = min_prefix if len(h) >= 3 else 2
    if len(h) < effective_min:
        return None
    for name in known_names:
        n = name.strip().lower()
        if n.startswith(h) and len(n) > len(h):
            print(f"[VOICE_NODE] Prefix rescue: '{heard}' → '{name}'")
            return name
    return None


def _soundex_match(heard: str, known_names: list):
    h = heard.strip().lower()
    if len(h) < 2:
        return None
    h_sdx = _soundex(h)
    candidates = []
    for name in known_names:
        n_sdx = _soundex(name)
        if h_sdx == n_sdx:
            candidates.append((name, 3))
        elif h_sdx[0] == n_sdx[0] and h_sdx[1] == n_sdx[1]:
            candidates.append((name, 2))
        elif h_sdx[0] == n_sdx[0] and len(h) <= 3:
            candidates.append((name, 1))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    best_name, best_score = candidates[0]
    if best_score >= 1:
        print(f"[VOICE_NODE] Soundex rescue: '{heard}' → '{best_name}' "
              f"(h={h_sdx} n={_soundex(best_name)} score={best_score})")
        return best_name
    return None


def jaro_winkler(s1: str, s2: str) -> float:
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(0, max(len1, len2) // 2 - 1)
    s2_matched = [False] * len2
    matches = transpositions = last_match = 0
    for i in range(len1):
        for j in range(max(0, i - match_dist), min(i + match_dist + 1, len2)):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s2_matched[j] = True
            matches += 1
            if j < last_match:
                transpositions += 1
            last_match = j
            break
    if matches == 0:
        return 0.0
    jaro   = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3.0
    prefix = sum(1 for i in range(min(4, len1, len2)) if s1[i] == s2[i])
    return jaro + prefix * 0.1 * (1.0 - jaro)


def rescue_name(heard: str, known_names: list) -> str:
    """Full 3-pass phonetic rescue. Returns best match or heard as-is."""
    if not known_names:
        return heard.strip()
    h     = heard.strip()
    h_low = h.lower()

    # 1. Exact
    for name in known_names:
        if name.lower() == h_low:
            return name

    # Arabic: no phonetic rescue, keep as-is
    if _is_arabic(heard):
        return h

    # 2. Prefix
    r = _prefix_match(h, known_names)
    if r:
        return r

    # 3. Soundex
    r = _soundex_match(h, known_names)
    if r:
        return r

    # 4. Jaro-Winkler
    best_name, best_score = h.title(), 0.0
    for name in known_names:
        if _is_arabic(name):
            continue
        score = jaro_winkler(h_low, name.lower())
        if score > best_score:
            best_score, best_name = score, name
    if best_score >= 0.75:
        print(f"[VOICE_NODE] JW rescue: '{heard}' → '{best_name}' ({best_score:.2f})")
        return best_name

    return h.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_arabic(text: str) -> bool:
    return any('\u0600' <= c <= '\u06FF' for c in text)


def _script_of(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin  = sum(1 for c in text if c.isascii() and c.isalpha())
    other  = sum(1 for c in text
                 if c.isalpha() and not c.isascii()
                 and not ('\u0600' <= c <= '\u06FF'))
    total  = arabic + latin + other
    if total == 0:
        return 'latin'
    if other / total > 0.3:
        return 'other'
    if arabic / total > 0.5:
        return 'arabic'
    return 'latin'


_COMMON_WORDS = {
    'where','what','who','when','why','how','the','this','that','here','there',
    'yes','no','not','okay','ok','sure','please','hello','hi','hey','bye',
    'good','bad','sorry','thank','thanks','can','could','would','should','will',
    'may','might','have','has','need','want','like','look','find','help','see',
    'come','go','know','think','say','tell','ask','just','also','back','still',
    'then','than','with','from','into','and','but','for','are','was','were',
    'been','being','its','our','your','their','his','her','him','them','they',
    'you','she','inc','ltd','vous','moi','toi','lui','elle','nous','ils','elles',
    'des','pour','avec','dans','sur','sous','vers','chez','par','les','mais',
    'donc','puis','bien','tres','tout','rien','une','est','aux','ces','mes',
    'ses','tes',
}

NAME_TRIGGERS = [
    "je m'appelle ", "mon nom est ", "my name is ",
    "call me ", "i am called ", "ils m'appellent ",
]


def _is_plausible_name_word(word: str) -> bool:
    if "'" in word or "\u2019" in word:
        return False
    alpha = ''.join(c for c in word if c.isalpha())
    if len(alpha) < 2:
        return False
    if alpha.lower() in _COMMON_WORDS:
        return False
    return True


def extract_full_name(text: str):
    if _is_arabic(text):
        c = text.strip()
        return c if len(c) >= 2 else None
    words = text.strip().split()
    parts = []
    for word in words:
        if _is_plausible_name_word(word):
            clean = ''.join(c for c in word if c.isalpha() or c == '-').capitalize()
            if len(clean) >= 2:
                parts.append(clean)
        else:
            break
    if not parts:
        return None
    full = ' '.join(parts)
    return full if len(full) <= 40 else None


# ─────────────────────────────────────────────────────────────────────────────
# VoiceNode
# ─────────────────────────────────────────────────────────────────────────────
class VoiceNode:

    ROBOT_PHRASES = [
        "how may i assist you", "how can i assist you", "i am here to help",
        "what do you need", "nice to meet you", "welcome back", "goodbye",
        "have a wonderful day", "how may i help", "what can i do for you",
        "i don't think we've met", "what's your name", "could you please introduce",
        "great to meet you", "great to see you", "your third visit",
        "i heard", "is that correct", "type your name", "text box",
        "what topic interests", "fun facts", "perfect i'll share",
        "got it i'll find", "say your name again", "didn't catch that",
        "please say your name", "please repeat", "having trouble hearing",
        "type it in the text box",
        # Extra fragments caught from gTTS echo
        "hello i don", "i don't think", "don't think we", "think we've met",
        "we've met before", "met before", "before what", "your name",
        "hello i", "did you know", "fun fact", "welcome back",
        "great to see", "brain consolidates", "computer bug",
    ]

    def __init__(self, command_queue, face_recognition_node=None, wake_word_mode=False):
        self.command_queue         = command_queue
        self.face_recognition_node = face_recognition_node
        self.wake_word_mode        = wake_word_mode
        self.running               = False
        self.sound_node            = None
        self.enabled               = SR_AVAILABLE and PYAUDIO_AVAILABLE

        self._rec_normal = sr.Recognizer() if SR_AVAILABLE else None
        self._rec_name   = sr.Recognizer() if SR_AVAILABLE else None

        if self._rec_normal:
            self._rec_normal.energy_threshold         = 300
            self._rec_normal.dynamic_energy_threshold = True
            self._rec_normal.pause_threshold          = 0.6
            self._rec_normal.phrase_threshold         = 0.3
            self._rec_normal.non_speaking_duration    = 0.3

        if self._rec_name:
            self._rec_name.energy_threshold         = 180
            self._rec_name.dynamic_energy_threshold = False
            self._rec_name.pause_threshold          = 1.8
            self._rec_name.phrase_threshold         = 0.1
            self._rec_name.non_speaking_duration    = 1.5

        self.microphone = None
        self.whisper_model = None

        self._active       = not wake_word_mode
        self._active_until = 0
        self._active_secs  = 10
        self._active_lock  = threading.Lock()

        self._robot_speaking        = False
        self._robot_speaking_until  = 0
        self._last_robot_speech_end = 0

        self._waiting_confirm   = False
        self._pending_name      = None
        self._confirm_deadline  = 0
        self._name_saved_by_gui = False
        self.NAME_CONFIRM_TIMEOUT = 10

        self.cooldown_after_robot          = 6.0
        self._last_robot_response          = ""
        self._last_robot_response_time     = 0

        self._preference_mode         = False
        self._preference_tts_done_at  = 0.0

        mode = "wake-word" if wake_word_mode else "continuous"
        print(f"[VOICE_NODE] Initialized — {mode} | enabled={self.enabled}")

    # ── Public API ────────────────────────────────────────────────────

    def cancel_pending_name(self):
        if self._waiting_confirm:
            print(f"[VOICE_NODE] GUI name received — discarding '{self._pending_name}'")
        self._waiting_confirm   = False
        self._pending_name      = None
        self._confirm_deadline  = 0
        self._name_saved_by_gui = True

    def set_preference_mode(self, tts_text: str):
        word_count = len(tts_text.split())
        duration = 0.5 + word_count * 0.45 + 0.5
        self._preference_mode        = True
        self._preference_tts_done_at = time.time() + duration
        print(f"[VOICE_NODE] Preference mode ON — mic opens in ~{duration:.1f}s")

    def set_sound_node(self, sn):
        self.sound_node = sn
        print("[VOICE_NODE] Sound node connected")

    def activate_wake_word(self):
        with self._active_lock:
            self._active       = True
            self._active_until = time.time() + self._active_secs

    # ── Whisper ───────────────────────────────────────────────────────

    def _load_whisper(self):
        if not WHISPER_AVAILABLE:
            return
        try:
            print(f"[VOICE_NODE] Loading Whisper 'small' on {DEVICE}...")
            self.whisper_model = WhisperModel(
                "small", device=DEVICE, compute_type=COMPUTE_TYPE,
                cpu_threads=5 if DEVICE == "cpu" else 3, num_workers=2,
            )
            print("[VOICE_NODE] Whisper ready")
        except Exception as e:
            print(f"[VOICE_NODE] Whisper load failed: {e}")

    # ── Transcription ─────────────────────────────────────────────────

    def _transcribe_google(self, audio, name_mode=False):
        rec = self._rec_name if name_mode else self._rec_normal
        if rec is None:
            return None
        langs = ["ar-TN", "fr-FR", "en-US"] if name_mode else ["en-US", "fr-FR"]
        for lang in langs:
            try:
                t = rec.recognize_google(audio, language=lang)
                if t and t.strip():
                    print(f"[VOICE_NODE] Google [{lang}] → {t!r}")
                    return t.strip()
            except Exception:
                continue
        return None

    def _transcribe_whisper(self, audio, name_mode=False):
        if self.whisper_model is None:
            return None
        try:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            np_audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

            initial_prompt = None
            if name_mode:
                known = self._get_known_names()
                if known:
                    initial_prompt = ("The visitor is saying their name. "
                                      "Known names: " + ", ".join(known) + ".")

            segs, info = self.whisper_model.transcribe(
                np_audio,
                language=None,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.3 if name_mode else 0.5,
                    min_silence_duration_ms=200 if name_mode else 400,
                    max_speech_duration_s=8 if name_mode else 12,
                ),
                beam_size=5,
                best_of=5,
                patience=1.0 if name_mode else 0.8,
                temperature=0.0,
                suppress_tokens=[-1],
            )
            text = " ".join(s.text.strip() for s in segs if s.text.strip()).strip()
            if not text:
                return None

            min_prob = 0.70 if name_mode else 0.15
            if info.language_probability < min_prob:
                print(f"[VOICE_NODE] Whisper rejected — confidence {info.language_probability:.0%} "
                      f"< {min_prob:.0%} ({'name' if name_mode else 'conv'} mode): {text!r}")
                return None

            if _script_of(text) == 'other':
                print(f"[VOICE_NODE] Whisper rejected — wrong script ({info.language}): {text!r}")
                return None

            print(f"[VOICE_NODE] Whisper ({info.language} {info.language_probability:.0%})"
                  f"{'[NAME]' if name_mode else ''} → {text!r}")
            return text
        except Exception as e:
            print(f"[VOICE_NODE] Whisper: {e}")
            return None

    def _transcribe_both(self, audio, name_mode=False):
        g = self._transcribe_google(audio, name_mode)
        w = self._transcribe_whisper(audio, name_mode)

        if not g and not w:
            return ""
        if not g:
            return w
        if not w:
            return g
        if g.lower().strip() == w.lower().strip():
            return g

        if name_mode:
            gs = _script_of(g)
            ws = _script_of(w)
            if gs != ws:
                print(f"[VOICE_NODE] Script conflict: G={g!r}({gs}) W={w!r}({ws}) → Google wins")
                return g
            result = g if len(g) >= len(w) else w
            print(f"[VOICE_NODE] Vote: G={g!r} W={w!r} → {result!r}")
            return result

        return g

    # ── Name recording loop ───────────────────────────────────────────

    def _listen_for_name(self, max_attempts=3):
        known = self._get_known_names()

        for attempt in range(1, max_attempts + 1):
            if self._name_saved_by_gui:
                return None

            if attempt > 1:
                self._send_speak_command(
                    "I didn't catch that — please say your name again clearly.")
                t0 = time.time()
                while self._is_robot_speaking() and time.time() - t0 < 10.0:
                    time.sleep(0.1)
                time.sleep(1.0)

            print(f"[VOICE_NODE] Name attempt {attempt}/{max_attempts} "
                  f"(pause_thr={self._rec_name.pause_threshold}s)…")

            audio = self._listen_once(name_mode=True)
            if audio is None:
                print(f"[VOICE_NODE] No audio on attempt {attempt}")
                continue

            raw_text = self._transcribe_both(audio, name_mode=True)
            if not raw_text:
                print(f"[VOICE_NODE] STT empty on attempt {attempt}")
                continue

            candidate = None
            low = raw_text.lower()
            for trigger in NAME_TRIGGERS:
                if trigger in low:
                    after = low.split(trigger, 1)[1].strip()
                    candidate = extract_full_name(after)
                    break
            if not candidate:
                candidate = extract_full_name(raw_text)
            if not candidate:
                candidate = raw_text.strip()

            rescued = rescue_name(candidate, known)
            print(f"[VOICE_NODE] raw={raw_text!r} cand={candidate!r} rescued={rescued!r}")

            if len(rescued.replace(' ', '')) >= 3 or rescued in known:
                return rescued

            print(f"[VOICE_NODE] '{rescued}' still too short, retrying…")

        return None

    # ── Standard listen ───────────────────────────────────────────────

    def _listen_once(self, name_mode=False):
        if self._is_robot_speaking():
            time.sleep(0.15)
            return None
        rec     = self._rec_name if name_mode else self._rec_normal
        timeout = 8.0 if name_mode else 5.0
        limit   = 15.0 if name_mode else 8.0
        try:
            with self.microphone as source:
                return rec.listen(source, timeout=timeout, phrase_time_limit=limit)
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[VOICE_NODE] listen: {e}")
            return None

    # ── Robot detection ───────────────────────────────────────────────

    def _is_robot_speaking(self):
        if self._preference_mode:
            if time.time() >= self._preference_tts_done_at:
                return False
            return True
        if self._robot_speaking:
            return True
        if time.time() < self._robot_speaking_until:
            return True
        if self.sound_node and self.sound_node.is_currently_speaking():
            return True
        return False

    def _is_robot_echo(self, text):
        if not text:
            return True
        tl = text.lower().strip()
        for phrase in self.ROBOT_PHRASES:
            if phrase in tl or tl in phrase:
                print(f"[VOICE_NODE] Echo: '{text}'")
                return True
        if (self._last_robot_response
                and time.time() - self._last_robot_response_time < 4.0
                and not _is_arabic(text)):
            if jaro_winkler(text, self._last_robot_response) > 0.65:
                return True
        return False

    # ── Confirmation ──────────────────────────────────────────────────

    def _confirm_name(self, candidate):
        if not _is_arabic(candidate):
            candidate = candidate.title()
        self._waiting_confirm   = True
        self._pending_name      = candidate
        self._confirm_deadline  = time.time() + self.NAME_CONFIRM_TIMEOUT
        self._name_saved_by_gui = False
        msg = (f"I heard {candidate}. "
               f"If this is incorrect, please type your name in the text box.")
        print(f"[VOICE_NODE] Confirming '{candidate}', auto-save in {self.NAME_CONFIRM_TIMEOUT}s")
        self._send_speak_command(msg)

    def _check_pending_name_timeout(self):
        if not self._waiting_confirm:
            return
        if self._name_saved_by_gui:
            self._waiting_confirm = self._name_saved_by_gui = False
            self._pending_name    = None
            self._confirm_deadline = 0
            return
        if self._pending_name and time.time() >= self._confirm_deadline:
            print(f"[VOICE_NODE] Auto-saving '{self._pending_name}'")
            name = self._pending_name
            self._waiting_confirm  = False
            self._pending_name     = None
            self._confirm_deadline = 0
            self._save_name(name)

    def _save_name(self, name):
        if not name or len(name.strip()) < 2:
            return
        name = name.strip() if _is_arabic(name) else name.strip().title()
        print(f"[VOICE_NODE] SET_NAME → '{name}'")
        self._waiting_confirm = False
        self._pending_name    = None
        self.command_queue.put({'type': 'SET_NAME', 'name': name})

    # ── Text processing ───────────────────────────────────────────────

    def _process(self, text):
        if not text or len(text.strip()) < 1:
            return
        if self._is_robot_echo(text):
            return

        # ── FIX 2: if preference TTS is still playing, drop EVERYTHING ──
        # This catches Google's early partial result (e.g. 'Hist') that
        # arrives before the question has finished being spoken.
        if self._preference_mode and time.time() < self._preference_tts_done_at:
            print(f"[VOICE_NODE] Pref TTS still playing, dropping early result: '{text}'")
            return

        # In preference mode bypass the post-speech cooldown entirely
        if not self._preference_mode:
            if time.time() - self._last_robot_speech_end < self.cooldown_after_robot:
                print(f"[VOICE_NODE] Cooldown, skip: {text}")
                return

        if self._name_saved_by_gui:
            self._name_saved_by_gui = False
            return

        low = text.lower().strip()

        if self._waiting_confirm:
            yes = ['yes', 'correct', 'yep', 'yeah', 'ok', 'okay', 'oui', 'affirm']
            no  = ['no', 'nope', 'wrong', 'non', 'incorrect']
            if any(w in low for w in yes):
                n = self._pending_name
                self._waiting_confirm = False
                self._pending_name = self._confirm_deadline = None
                self._save_name(n)
                return
            if any(w in low for w in no):
                self._waiting_confirm = False
                self._pending_name = None
                self._confirm_deadline = 0
                self._send_speak_command("No problem — please say your name again.")
                return
            self._waiting_confirm  = False
            self._pending_name     = None
            self._confirm_deadline = 0

        if low in ('hey aria', 'aria', 'hey', 'area', ''):
            return

        for trigger in NAME_TRIGGERS:
            if trigger in low:
                after = low.split(trigger, 1)[1].strip()
                c = extract_full_name(after)
                if c and 2 <= len(c) <= 40:
                    r = rescue_name(c, self._get_known_names())
                    self._confirm_name(r)
                    return

        fr   = self.face_recognition_node
        gate = (fr is not None and hasattr(fr, 'is_waiting_for_name')
                and fr.is_waiting_for_name())
        if gate and 1 <= len(text.strip().split()) <= 4:
            c = extract_full_name(text)
            if c and 2 <= len(c) <= 40:
                r = rescue_name(c, self._get_known_names())
                self._confirm_name(r)
                return

        self.command_queue.put({'type': 'USER_MESSAGE', 'text': text})

    # ── Main loop ─────────────────────────────────────────────────────

    def _loop(self):
        consecutive_errors = 0
        while self.running:
            if not self._active:
                time.sleep(0.2)
                continue

            fr   = self.face_recognition_node
            gate = (fr is not None and hasattr(fr, 'is_waiting_for_name')
                    and fr.is_waiting_for_name())

            if gate and not self._waiting_confirm and not self._name_saved_by_gui:
                t0 = time.time()
                while self._is_robot_speaking() and time.time() - t0 < 12.0:
                    time.sleep(0.1)
                # Extra buffer — gTTS playback may still be finishing
                time.sleep(1.5)

                result = self._listen_for_name(max_attempts=3)

                if result:
                    self._confirm_name(result)
                elif not self._name_saved_by_gui:
                    self._send_speak_command(
                        "I'm having trouble hearing your name. "
                        "Could you please type it in the text box?")
                continue

            try:
                audio = self._listen_once(name_mode=False)
                if audio is None:
                    self._check_pending_name_timeout()
                    time.sleep(0.05)
                    continue
                text = self._transcribe_both(audio, name_mode=False)
                self._check_pending_name_timeout()
                if text:
                    consecutive_errors = 0
                    self._process(text)
                else:
                    time.sleep(0.06)
            except Exception as e:
                consecutive_errors += 1
                print(f"[VOICE_NODE] Loop error #{consecutive_errors}: {e}")
                time.sleep(4 if consecutive_errors > 6 else 0.7)
                if consecutive_errors > 6:
                    consecutive_errors = 0

    # ── Mic init ──────────────────────────────────────────────────────

    def _init_mic(self):
        if not self.enabled:
            return False
        try:
            print("[VOICE_NODE] Opening microphone…")
            self.microphone = sr.Microphone()
            print("[VOICE_NODE] Calibrating 2.5 s (stay quiet)…")
            with self.microphone as src:
                self._rec_normal.adjust_for_ambient_noise(src, duration=2.5)
                self._rec_name.energy_threshold = max(
                    150, self._rec_normal.energy_threshold * 0.7)
            print(f"[VOICE_NODE] Ready — "
                  f"conv={self._rec_normal.energy_threshold:.0f} "
                  f"name={self._rec_name.energy_threshold:.0f}")
            return True
        except Exception as e:
            print(f"[VOICE_NODE] Mic init failed: {e}")
            self.enabled = False
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    def _send_speak_command(self, text):
        if text and text.strip():
            self.command_queue.put({'type': 'ROBOT_SPEAK', 'text': text.strip()})

    def _get_known_names(self):
        if self.face_recognition_node is None:
            return []
        try:
            return [v['name'] for v in
                    getattr(self.face_recognition_node, 'visitors', {}).values()
                    if v.get('name') and not v['name'].startswith('Visitor_')]
        except Exception:
            return []

    # ── Lifecycle ─────────────────────────────────────────────────────

    def run(self):
        self.running = True
        if not self.enabled:
            print("[VOICE_NODE] Disabled")
            while self.running:
                time.sleep(1.5)
            return
        if not self._init_mic():
            print("[VOICE_NODE] Mic init failed")
            while self.running:
                time.sleep(1.5)
            return
        threading.Thread(target=self._load_whisper, daemon=True).start()
        print("[VOICE_NODE] Running — name mode uses dedicated slow recorder")
        try:
            self._loop()
        finally:
            self.shutdown()

    def shutdown(self):
        self.running         = False
        self._robot_speaking = False
        print("[VOICE_NODE] Stopped")