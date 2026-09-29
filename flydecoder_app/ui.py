"""Low-resolution Pygame interface and original pixel-art scene."""
from __future__ import annotations

import math
import queue
import random
import threading
import time
from pathlib import Path

import pygame

from .brain_adapter import ACTIVITY_NAMES, FlyBrainAdapter
from .features import extract_features
from .generator import examples
from .readout import ACTIONS, ActivityReadout
from .service import decode_with_retry, run_blind_evaluation
from .training import TrainingProgress, train_readout


VIRTUAL = (520, 390)
SCALE = 2
DISPLAY = (VIRTUAL[0] * SCALE, VIRTUAL[1] * SCALE)
MODEL_PATH = Path("out/decoder_reader.npz")
_TEXT_CANVAS = None
_TEXT_OVERLAYS = None

PALETTE = {
    "ink": (27, 35, 48), "deep": (30, 47, 57), "night": (43, 67, 72),
    "wall": (102, 112, 94), "window": (66, 102, 112), "moon": (201, 193, 147),
    "table": (155, 103, 63), "table_hi": (195, 138, 83), "wood": (115, 72, 54),
    "cream": (239, 222, 177), "paper": (220, 211, 177), "green": (130, 159, 112),
    "mint": (161, 196, 142), "red": (202, 111, 90), "pink": (217, 155, 147),
    "blue": (119, 164, 177), "darkblue": (56, 80, 105), "coffee": (77, 46, 35),
    "white": (243, 237, 216), "muted": (183, 198, 184), "gold": (230, 190, 96),
    "panel": (46, 61, 66), "panel2": (57, 74, 77), "outline": (25, 38, 45),
}


def _rect(surface, color, rect, width=0):
    pygame.draw.rect(surface, color, pygame.Rect(rect), width)


def _text(surface, font, value, x, y, color=None):
    ink = color or PALETTE["cream"]
    if surface is _TEXT_CANVAS and _TEXT_OVERLAYS is not None:
        _TEXT_OVERLAYS.append((font, str(value), x, y, ink))
    else:
        surface.blit(font.render(str(value), True, ink), (x, y))


def _fit_text(font, value, max_width, suffix="…"):
    """Keep interface copy inside its panel without clipping glyphs."""
    value = str(value)
    if font.size(value)[0] <= max_width:
        return value
    while value and font.size(value + suffix)[0] > max_width:
        value = value[:-1]
    return value + suffix if value else suffix


def _fit_suffix(font, value, max_width):
    """Show the end of long encoded rows, where the distinguishing bytes often are."""
    value = str(value)
    while value and font.size(value)[0] > max_width:
        value = value[1:]
    return value


def _bar(surface, x, y, width, value, color, height=4):
    _rect(surface, PALETTE["outline"], (x, y, width, height))
    _rect(surface, color, (x, y, max(1, int(width * max(0.0, min(1.0, float(value))))), height))


class DecoderWindow:
    def __init__(self, model_path: str | Path = MODEL_PATH, headless: bool = False,
                 demo_text: str = "SGVsbG8gZmx5IQ==", adapter=None, model=None):
        pygame.init()
        self.headless = headless
        self.display = pygame.display.set_mode(DISPLAY)
        pygame.display.set_caption("FlyDecoder · MaleCNS v1.0")
        pygame.key.start_text_input()
        try:
            pygame.scrap.init()
        except pygame.error:
            pass
        self.canvas = pygame.Surface(VIRTUAL)
        self.font = pygame.font.SysFont("Segoe UI", 10)
        self.small = pygame.font.SysFont("Segoe UI", 9)
        self.bold = pygame.font.SysFont("Segoe UI", 10, bold=True)
        self.display_fonts = {
            id(self.font): pygame.font.SysFont("Segoe UI", 20),
            id(self.small): pygame.font.SysFont("Segoe UI", 18),
            id(self.bold): pygame.font.SysFont("Segoe UI", 20, bold=True),
        }
        self.clock = pygame.time.Clock()
        self.messages: queue.Queue = queue.Queue()
        self.model_path = Path(model_path)
        self.adapter = adapter
        self.model = model
        self.ready = adapter is not None and model is not None
        self.busy = not self.ready
        self.running = True
        self.cancel = threading.Event()
        self.input_text = demo_text
        self.input_active = True
        self.feature_page = 0
        self.feature = extract_features(self.input_text)
        self.activity = None
        self.result = None
        self.metrics = None
        self.training = None
        self.activity_page = 0
        self.status = "Завантаження симуляції та MaleCNS…"
        self.action_state = "idle"
        self.action_started = time.time()
        self.last_coffee = time.time() + 16.0
        self.started = time.time()
        self.sparkles = random.Random(19)
        if not self.ready:
            self._start_worker(self._initialize)

    def _start_worker(self, target):
        threading.Thread(target=target, daemon=True).start()

    def _initialize(self):
        try:
            adapter = FlyBrainAdapter(device="cpu")
            if self.model_path.exists():
                model = ActivityReadout.load(self.model_path)
                if model.input_count != len(ACTIVITY_NAMES):
                    model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
            else:
                model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
            self.messages.put(("ready", adapter, model))
        except Exception as exc:
            self.messages.put(("error", f"Не вдалося завантажити мозок: {exc}"))

    def _begin_decode(self):
        if not self.ready or self.busy or not self.input_text.strip():
            return
        text = self.input_text
        self.feature = extract_features(text)
        self.result = None
        self.metrics = None
        self.busy = True
        self.status = "Муха нахиляється й аналізує рядок…"
        self._set_state("analyze")

        def task():
            try:
                # Only numerical features cross the adapter boundary. The original row remains here
                # for the selected decoder after the simulator has produced its activity readout.
                activity = self.adapter.activity(self.feature.values.copy())
                result = decode_with_retry(text, activity, self.model)
                self.messages.put(("decoded", activity, result))
            except Exception as exc:
                self.messages.put(("error", f"Помилка симуляції: {exc}"))

        self._start_worker(task)

    def _begin_training(self, episodes: int = 64):
        if not self.ready or self.busy:
            return
        self.busy = True
        self.training = None
        self.status = "Навчання: муха обирає, оцінювач винагороджує точні байти…"
        self._set_state("type")

        def update(progress: TrainingProgress):
            self.messages.put(("train_progress", progress))

        def task():
            try:
                result = train_readout(self.adapter, self.model, episodes, seed=9301,
                                       save_path=self.model_path, progress=update, cancel=self.cancel)
                self.messages.put(("trained", result))
            except Exception as exc:
                self.messages.put(("error", f"Помилка навчання: {exc}"))

        self._start_worker(task)

    def _begin_blind(self, count: int = 32):
        if not self.ready or self.busy:
            return
        self.busy = True
        self.metrics = None
        self.status = "Сліпа перевірка на нових рядках…"
        self._set_state("analyze")

        def task():
            try:
                metrics = run_blind_evaluation(self.adapter, self.model, count=count, seed=8041)
                self.messages.put(("blind", metrics))
            except Exception as exc:
                self.messages.put(("error", f"Помилка сліпої перевірки: {exc}"))

        self._start_worker(task)

    def _set_state(self, state: str):
        self.action_state = state
        self.action_started = time.time()

    def _handle_messages(self):
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return
            kind = message[0]
            if kind == "ready":
                self.adapter, self.model = message[1], message[2]
                self.ready = True
                self.busy = False
                self.status = f"Готово · CPU · ваг зчитувача: {self.model.updates}"
            elif kind == "decoded":
                self.activity, self.result = message[1], message[2]
                self.busy = False
                if self.result.output is not None:
                    self.status = self.result.message
                    self._set_state("success" if len(self.result.attempts) <= 1 else "retry")
                else:
                    self.status = self.result.message
                    self._set_state("failure")
            elif kind == "train_progress":
                self.training = message[1]
                self.status = (f"Навчання {self.training.completed}/{self.training.total} · "
                               f"точність винагороди {self.training.recent_accuracy:.0%}")
            elif kind == "trained":
                self.training = message[1]
                self.busy = False
                self.status = f"Навчання збережено · {self.model.updates} рішень"
                self._set_state("success")
            elif kind == "blind":
                self.metrics = message[1]
                self.busy = False
                self.status = (f"Сліпа перевірка · перша спроба {self.metrics['top1_accuracy']:.0%}, "
                               f"з повторами {self.metrics['retry_accuracy']:.0%}")
                self._set_state("select")
            elif kind == "error":
                self.status = message[1]
                self.busy = False
                self._set_state("failure")

    def _events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.TEXTINPUT and self.input_active and not self.busy:
                self.input_text = (self.input_text + event.text)[-12000:]
                self.feature = extract_features(self.input_text)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_RETURN and not (event.mod & pygame.KMOD_SHIFT):
                    self._begin_decode()
                elif event.key == pygame.K_BACKSPACE and not self.busy:
                    self.input_text = self.input_text[:-1]
                    self.feature = extract_features(self.input_text)
                elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL and not self.busy:
                    try:
                        pasted = pygame.scrap.get(pygame.SCRAP_TEXT)
                        if pasted:
                            self.input_text = (self.input_text + pasted.decode("utf-8", "replace").rstrip("\x00"))[-12000:]
                            self.feature = extract_features(self.input_text)
                    except (pygame.error, AttributeError):
                        pass
                elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    delta = 1 if event.key == pygame.K_PAGEDOWN else -1
                    if event.mod & pygame.KMOD_SHIFT:
                        self.activity_page = max(0, self.activity_page + delta)
                    else:
                        self.feature_page = max(0, self.feature_page + delta)
                elif event.key == pygame.K_TAB:
                    self.input_active = not self.input_active
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos[0] // SCALE, event.pos[1] // SCALE
                if 178 <= y < 205:
                    self.input_active = True
                elif 209 <= y < 232:
                    if x < 105:
                        self._begin_decode()
                    elif x < 207:
                        self._begin_training()
                    elif x < 339:
                        self._begin_blind()
                    elif x < 431 and not self.busy:
                        self.input_text = ""
                        self.feature = extract_features("")
                        self.result = None
                        self.input_active = True
                elif 291 <= y < 384 and x < 266:
                    self.feature_page += 1
                elif 291 <= y < 384 and x >= 272:
                    self.activity_page += 1
        if self.input_active:
            pygame.key.set_text_input_rect(pygame.Rect(18 * SCALE, 182 * SCALE, 482 * SCALE, 20 * SCALE))

    def _scene(self):
        s = self.canvas
        p = PALETTE
        _rect(s, p["night"], (0, 0, 520, 150))
        _rect(s, p["wall"], (0, 89, 520, 28))
        _rect(s, p["deep"], (16, 19, 88, 57))
        _rect(s, p["window"], (20, 23, 80, 49))
        _rect(s, p["ink"], (57, 23, 4, 49)); _rect(s, p["ink"], (20, 45, 80, 4))
        _rect(s, p["moon"], (77, 30, 8, 8)); _rect(s, p["blue"], (27, 54, 13, 9))
        # Warm lamp and its pool of light.
        _rect(s, p["gold"], (444, 11, 18, 3)); _rect(s, p["gold"], (451, 14, 4, 19))
        _rect(s, (133, 112, 79), (418, 31, 70, 4)); _rect(s, p["cream"], (446, 32, 18, 3))
        _rect(s, p["wood"], (0, 119, 520, 31)); _rect(s, p["table_hi"], (0, 115, 520, 5))
        for x in range(0, 520, 43):
            _rect(s, (174, 119, 75), (x, 132 + (x % 3), 31, 1))
        # Chair and legs.
        _rect(s, p["outline"], (93, 82, 5, 36)); _rect(s, p["outline"], (126, 82, 5, 36))
        _rect(s, p["table_hi"], (88, 78, 49, 5)); _rect(s, p["wood"], (88, 83, 46, 4))
        # Laptop, with tiny colored codec tiles on the screen.
        _rect(s, p["outline"], (280, 73, 94, 45)); _rect(s, p["darkblue"], (284, 77, 86, 37))
        _rect(s, p["blue"], (290, 82, 74, 25))
        chosen = self.result.first_action if self.result else "base64"
        tool_colors = {"base64": p["gold"], "hex": p["pink"], "binary": p["mint"], "url": p["blue"], "skip": p["muted"]}
        for i, action in enumerate(ACTIONS):
            color = tool_colors[action]
            x = 291 + i * 14
            _rect(s, color if action == chosen else (83, 112, 115), (x, 86, 9, 7))
            _rect(s, p["ink"], (x + 2, 88, 5, 1))
        _rect(s, p["outline"], (272, 117, 110, 4)); _rect(s, (115, 134, 137), (279, 114, 97, 3))
        for i in range(9):
            _rect(s, (188, 178, 147), (284 + i * 10, 115, 6, 1))
        # Coffee cup and steam.
        cup_shift = -6 if self.action_state == "sip" else 0
        cup_x = 245 + cup_shift
        _rect(s, p["cream"], (cup_x, 99, 19, 17)); _rect(s, p["coffee"], (cup_x + 2, 101, 15, 3))
        _rect(s, p["cream"], (cup_x + 18, 103, 6, 7)); _rect(s, p["table_hi"], (cup_x + 20, 105, 2, 3))
        if self.action_state == "sip" or self.sparkles.random() < 0.015:
            _rect(s, p["muted"], (cup_x + 6, 93, 1, 4)); _rect(s, p["muted"], (cup_x + 12, 91, 1, 5))
        # Pixel fly: segmented body, veined wings, six legs, eyes, and antennae.
        elapsed = time.time() - self.action_started
        blink = (self.action_state == "idle" and int((time.time() - self.started) * 1.4) % 9 == 0) or (self.action_state == "blink")
        lean = 5 if self.action_state in ("analyze", "type", "select", "decode", "retry") else (3 if self.action_state == "sip" else 0)
        bounce = int(math.sin(elapsed * 7) * 1.2) if self.action_state in ("type", "analyze") else 0
        fly = pygame.Surface((76, 64), pygame.SRCALPHA)
        center_x, center_y = 31 + lean, 32 + bounce
        wing_color = (161, 194, 183, 185)
        pygame.draw.ellipse(fly, wing_color, (13 + lean // 3, 8, 25, 13))
        pygame.draw.ellipse(fly, (148, 181, 171, 175), (28 + lean // 3, 4, 25, 14))
        pygame.draw.line(fly, (222, 231, 210, 220), (18, 18), (34, 12), 1)
        pygame.draw.line(fly, (222, 231, 210, 220), (34, 17), (45, 10), 1)

        # Three visible body sections give the insect a clear head, thorax, and abdomen.
        pygame.draw.ellipse(fly, p["outline"], (7, center_y - 5, 25, 14))
        pygame.draw.ellipse(fly, (76, 58, 48), (10, center_y - 3, 20, 10))
        for x in (15, 21, 26):
            pygame.draw.line(fly, (173, 99, 63), (x, center_y - 2), (x + 1, center_y + 5), 1)
        pygame.draw.ellipse(fly, (97, 70, 52), (27, center_y - 9, 17, 18))
        pygame.draw.ellipse(fly, (144, 78, 55), (37, center_y - 11, 17, 16))
        pygame.draw.ellipse(fly, (207, 65, 62), (43, center_y - 9, 8, 8))
        if blink:
            pygame.draw.line(fly, p["outline"], (44, center_y - 5), (50, center_y - 5), 2)
        else:
            pygame.draw.circle(fly, p["white"], (46, center_y - 7), 2)
        pygame.draw.line(fly, p["outline"], (46, center_y - 10), (49, center_y - 17), 1)
        pygame.draw.line(fly, p["outline"], (49, center_y - 17), (54, center_y - 19), 1)
        pygame.draw.line(fly, p["outline"], (50, center_y - 9), (56, center_y - 15), 1)
        pygame.draw.line(fly, p["outline"], (56, center_y - 15), (62, center_y - 15), 1)
        # A fine proboscis reaches toward the cup; the legs remain visibly separate.
        pygame.draw.lines(fly, p["outline"], False,
                          [(51, center_y + 1), (57, center_y + 4), (63, center_y + 5)], 1)
        tap = int(elapsed * 8) % 2 if self.action_state == "type" else 0
        legs = (
            [(30, center_y + 4), (25, center_y + 11), (19, center_y + 17)],
            [(32, center_y + 5), (30, center_y + 13), (29, center_y + 19 - tap)],
            [(35, center_y + 5), (39, center_y + 12), (40, center_y + 19 + tap)],
            [(39, center_y + 3), (47, center_y + 9), (52, center_y + 16)],
            [(28, center_y + 3), (20, center_y + 10), (14, center_y + 15)],
            [(37, center_y + 4), (44, center_y + 12), (47, center_y + 18)],
        )
        for points in legs:
            pygame.draw.lines(fly, p["outline"], False, points, 1)
        if self.action_state in ("success", "retry"):
            _rect(fly, p["mint"], (56, 7, 5, 5)); _rect(fly, p["white"], (58, 5, 2, 2))
        elif self.action_state == "failure":
            _rect(fly, p["red"], (56, 7, 5, 5))
        elif self.action_state == "select":
            _rect(fly, p["gold"], (55, center_y + 9, 5, 5))
        sprite = pygame.transform.scale(fly, (95, 80))
        s.blit(sprite, (int(203 - 31 * 1.25 - lean / 2), int(92 - 32 * 1.25 - bounce / 2)))

    def _draw_panel(self):
        s = self.canvas
        p = PALETTE
        _rect(s, p["deep"], (0, 150, 520, 240))
        _rect(s, p["panel"], (0, 150, 520, 13))
        _text(s, self.bold, _fit_text(self.bold, self.status, 352), 9, 151,
              p["mint"] if self.ready else p["gold"])
        _text(s, self.small, "CPU · MaleCNS CC BY 4.0", 374, 152, p["muted"])
        _text(s, self.bold, "РЯДОК ДЛЯ ДЕКОДУВАННЯ", 9, 165)
        field_color = p["green"] if self.input_active else p["panel2"]
        _rect(s, p["outline"], (8, 178, 504, 27)); _rect(s, field_color, (9, 179, 502, 25))
        shown = _fit_suffix(self.font, self.input_text, 476)
        _text(s, self.font, shown or "Введіть або вставте рядок…", 14, 184, p["ink"])
        buttons = ((8, 209, 96, "ДЕКОДУВАТИ"), (109, 209, 96, "НАВЧАТИ"),
                   (210, 209, 125, "СЛІПА ПЕРЕВІРКА"), (340, 209, 88, "ОЧИСТИТИ"))
        for idx, (x, y, w, label) in enumerate(buttons):
            bg = p["table_hi"] if idx == 0 else p["panel2"]
            if self.busy and idx < 3:
                bg = (74, 82, 72)
            _rect(s, p["outline"], (x, y, w, 23)); _rect(s, bg, (x + 1, y + 1, w - 2, 21))
            _text(s, self.bold, label, x + 5, y + 7, p["ink"] if idx == 0 else p["cream"])
        if self.training:
            progress = self.training.completed / max(1, self.training.total)
            _bar(s, 435, 215, 77, progress, p["mint"], 5)
            _text(s, self.small, f"{self.training.recent_accuracy:.0%} точн.", 435, 222, p["muted"])
        elif self.model:
            _text(s, self.small, f"ваг {self.model.updates}", 444, 215, p["muted"])

        _rect(s, p["panel"], (8, 235, 258, 53)); _rect(s, p["outline"], (8, 235, 258, 53), 1)
        _text(s, self.bold, "РЕЗУЛЬТАТ", 14, 239, p["gold"])
        if self.result and self.result.output is not None:
            try:
                result_text = self.result.output.decode("utf-8", "strict")
            except UnicodeDecodeError:
                result_text = self.result.output.hex(" ")
            _text(s, self.font, _fit_text(self.font, result_text, 240), 14, 251, p["white"])
            first = self.result.first_action
            final = self.result.attempts[-1].action if self.result.attempts else first
            _text(s, self.small, f"перший: {first}   обрано: {final}", 14, 267, p["mint"])
        elif self.result:
            _text(s, self.font, _fit_text(self.font, self.result.message, 240), 14, 252, p["red"])
            _text(s, self.small, f"спроб: {len(self.result.attempts)} · дія: {self.result.first_action}", 14, 269, p["muted"])
        else:
            _text(s, self.font, "Введіть рядок і натисніть Enter.", 14, 255, p["muted"])
            _text(s, self.small, "Текст передається лише вибраному декодеру.", 14, 270, p["muted"])

        _rect(s, p["panel"], (272, 235, 240, 53)); _rect(s, p["outline"], (272, 235, 240, 53), 1)
        _text(s, self.bold, "ОЦІНКИ ДІЙ ЗЧИТУВАЧА", 278, 239, p["gold"])
        probs = self.model.probabilities(self.activity) if self.model and self.activity is not None else [0.2] * 5
        colors = (p["gold"], p["pink"], p["mint"], p["blue"], p["muted"])
        for i, (name, score) in enumerate(zip(ACTIONS, probs)):
            col, row = i % 3, i // 3
            x, y = 278 + col * 76, 251 + row * 17
            _text(s, self.small, f"{name:6s} {float(score):.0%}", x, y, colors[i])
            _bar(s, x, y + 13, 64, float(score), colors[i], 4)

        _rect(s, p["panel"], (8, 293, 258, 91)); _rect(s, p["outline"], (8, 293, 258, 91), 1)
        _text(s, self.bold, "ЧИСЛОВІ ОЗНАКИ", 14, 297, p["gold"])
        names, values = self.feature.names, self.feature.values
        page_size = 5
        start = (self.feature_page * page_size) % len(names)
        shown_indices = [(start + i) % len(names) for i in range(page_size)]
        for row, index in enumerate(shown_indices):
            y = 310 + row * 12
            _text(s, self.small, _fit_text(self.small, names[index], 125), 14, y, p["cream"])
            _bar(s, 147, y + 4, 78, float(values[index]), p["blue"], 5)
            _text(s, self.small, f"{float(values[index]):.2f}", 231, y, p["muted"])
        _text(s, self.small, _fit_text(self.small,
              f"{self.feature.summary['length']} символів · mod 4={self.feature.summary['mod4']} · mod 8={self.feature.summary['mod8']}", 244),
              14, 370, p["muted"])

        _rect(s, p["panel"], (272, 293, 240, 91)); _rect(s, p["outline"], (272, 293, 240, 91), 1)
        activity_pages = max(1, (len(ACTIVITY_NAMES) + 4) // 5)
        _text(s, self.bold, f"АКТИВНІ НЕЙРОННІ КАНАЛИ · {self.activity_page % activity_pages + 1}/{activity_pages}", 278, 297, p["gold"])
        if self.activity is None:
            activity = [0.0] * len(ACTIVITY_NAMES)
        else:
            activity = self.activity
        activity_start = (self.activity_page % activity_pages) * 5
        for slot, index in enumerate(range(activity_start, min(activity_start + 5, len(ACTIVITY_NAMES)))):
            name, value = ACTIVITY_NAMES[index], activity[index]
            y = 309 + slot * 12
            if name.startswith("optic_band_"):
                band = int(name.split("_")[2]) + 1
                group = f"Смуга {band:02d}"
            else:
                prefix = name.rsplit("_", 1)[0]
                group = {
                    "visual_projection": "Проекція",
                    "central": "Центр",
                    "mushroom_body": "Грибне тіло",
                    "descending": "Низхідні",
                    "vnc": "VNC",
                }[prefix]
            metric = "активність" if name.endswith("_mean") else "клітини"
            label = _fit_text(self.small, f"{group} · {metric}", 105)
            _text(s, self.small, label, 278, y, p["cream"])
            _bar(s, 390, y + 4, 112, float(value), p["mint"], 5)
        _text(s, self.small, "Shift+PgDn — наступні канали", 278, 372, p["muted"])

    def _draw(self):
        global _TEXT_CANVAS, _TEXT_OVERLAYS
        overlays = []
        _TEXT_CANVAS, _TEXT_OVERLAYS = self.canvas, overlays
        try:
            self._scene()
            self._draw_panel()
        finally:
            _TEXT_CANVAS, _TEXT_OVERLAYS = None, None
        pygame.transform.scale(self.canvas, DISPLAY, self.display)
        for font, value, x, y, color in overlays:
            glyphs = self.display_fonts[id(font)].render(value, True, color)
            self.display.blit(glyphs, (x * SCALE, y * SCALE))
        pygame.display.flip()

    def _animation(self):
        now = time.time()
        age = now - self.action_started
        if self.busy and self.action_state == "analyze" and age > 0.8:
            self._set_state("type")
        elif self.busy and self.action_state == "type" and age > 0.7:
            self._set_state("select")
        elif self.busy and self.action_state == "select" and age > 0.7:
            self._set_state("decode")
        elif not self.busy and self.action_state in ("success", "failure", "retry", "select") and age > 5:
            self._set_state("idle")
        elif not self.busy and self.action_state == "idle" and now > self.last_coffee:
            self.last_coffee = now + self.sparkles.uniform(20, 38)
            self._set_state("sip")
        elif not self.busy and self.action_state == "sip" and age > 2.4:
            self._set_state("idle")

    def run(self, max_seconds: float | None = None):
        start = time.time()
        while self.running:
            self._handle_messages()
            self._events()
            self._animation()
            self._draw()
            self.clock.tick(30)
            if max_seconds is not None and time.time() - start >= max_seconds:
                break
        self.cancel.set()
        pygame.key.stop_text_input()
        pygame.quit()
