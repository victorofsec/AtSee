from __future__ import annotations

import asyncio
import base64
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TTS_CLI_FLAG = "--speak"
DEFAULT_TTS_VOICE = "fr-FR-DeniseNeural"
DEFAULT_TTS_RATE = "+0%"


def normalize_speech_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_#`>]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_audio_player_command() -> list[str] | None:
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    if shutil.which("mpg123"):
        return ["mpg123", "-q"]
    return None


async def synthesize_neural_speech(text: str, voice: str, rate: str, output_path: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Module manquant: edge-tts. Lance: pip install -r requirements.txt") from exc

    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(output_path))


def play_audio_file(path: Path) -> int:
    player_command = find_audio_player_command()
    if not player_command:
        raise RuntimeError("Aucun lecteur MP3 trouve. Installe ffmpeg/ffplay ou mpg123.")

    player = subprocess.Popen(player_command + [str(path)])
    try:
        return player.wait()
    except KeyboardInterrupt:
        player.terminate()
        raise


def run_speech_cli(argv: list[str]) -> int:
    voice = DEFAULT_TTS_VOICE
    rate = DEFAULT_TTS_RATE
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg == "--voice" and index + 1 < len(argv):
            voice = argv[index + 1]
            index += 2
            continue
        if arg == "--rate" and index + 1 < len(argv):
            rate = argv[index + 1]
            index += 2
            continue
        index += 1

    text = sys.stdin.read()
    speech_text = normalize_speech_text(text)
    if not speech_text:
        return 0

    with tempfile.TemporaryDirectory(prefix="atsee-tts-") as temp_dir:
        output_path = Path(temp_dir) / "speech.mp3"
        asyncio.run(synthesize_neural_speech(speech_text, voice, rate, output_path))
        return play_audio_file(output_path)


if TTS_CLI_FLAG in sys.argv:
    raise SystemExit(run_speech_cli(sys.argv[1:]))

try:
    import cv2
except ImportError:
    print("Module manquant: opencv-python. Lance: pip install -r requirements.txt")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import hand_landmarker
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
except ImportError:
    print("Module manquant: mediapipe. Lance: pip install -r requirements.txt")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Module manquant: requests. Lance: pip install -r requirements.txt")
    sys.exit(1)


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
WINDOW_TITLE = "Atsee - capture par geste"

DEFAULT_MODEL = "openai/gpt-4.1"
DEFAULT_ENDPOINT = "https://models.github.ai/inference/chat/completions"
DEFAULT_API_VERSION = "2026-03-10"
DEFAULT_HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"
DEFAULT_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

INDEX_TIP = 8
MIDDLE_TIP = 12
TOKEN_KEYS = ("GITHUB_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS
)

SYSTEM_PROMPT = (
    "Tu es un assistant de vision. Si l'image contient une question, un exercice, "
    "une consigne ou un probleme lisible, reponds directement a ce qui est demande "
    "sans decrire la scene et sans paraphraser l'enonce. Si plusieurs questions "
    "sont visibles, reponds dans l'ordre. Si aucune question ou consigne n'est "
    "visible, decris l'image en francais, clairement et concisement. Ignore "
    "totalement les doigts, la main ou le geste de capture visibles au premier "
    "plan. Ne les mentionne pas, sauf si l'image ne montre rien d'autre "
    "d'important."
)

USER_PROMPT = (
    "Lis l'image. S'il y a une question ou une consigne visible, reponds-y "
    "directement sans decrire la scene. Sinon, decris l'image. Ignore mes doigts "
    "ou ma main au premier plan."
)


@dataclass(frozen=True)
class Config:
    token: str
    model: str
    endpoint: str
    api_version: str
    camera_index: int
    hold_frames: int
    max_image_side: int
    jpeg_quality: int
    hand_model_path: Path
    hand_model_url: str
    tts_voice: str
    tts_rate: str


class SpeechPlayer:
    def __init__(self, voice: str, rate: str) -> None:
        self.voice = voice
        self.rate = rate
        self.process: subprocess.Popen[str] | None = None
        self.paused = False
        self.last_error: str | None = None
        self.available = bool(find_audio_player_command())

        if not self.available:
            self.last_error = "Audio indisponible: installe ffplay ou mpg123."

    def _refresh(self) -> None:
        if self.process and self.process.poll() is not None:
            if self.process.returncode and self.process.returncode < 0:
                self.last_error = "Lecture audio interrompue."
            elif self.process.returncode:
                self.last_error = f"Audio IA indisponible (code {self.process.returncode})."
            self.process = None
            self.paused = False

    def speak(self, text: str) -> None:
        self.stop()
        speech_text = normalize_speech_text(text)
        if not speech_text:
            return

        if not self.available:
            return

        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    TTS_CLI_FLAG,
                    "--voice",
                    self.voice,
                    "--rate",
                    self.rate,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            if self.process.stdin is None:
                raise RuntimeError("stdin TTS indisponible.")
            self.process.stdin.write(speech_text)
            self.process.stdin.close()
            self.paused = False
            self.last_error = None
        except Exception as exc:
            self.stop()
            self.last_error = f"Audio indisponible: {exc}"

    def toggle_pause(self) -> None:
        self._refresh()
        if not self.process:
            return

        if not hasattr(signal, "SIGSTOP") or not hasattr(signal, "SIGCONT"):
            self.last_error = "Pause audio indisponible sur ce systeme."
            return

        try:
            target_signal = signal.SIGCONT if self.paused else signal.SIGSTOP
            if hasattr(os, "killpg"):
                os.killpg(self.process.pid, target_signal)
            else:
                os.kill(self.process.pid, target_signal)
            self.paused = not self.paused
        except OSError as exc:
            self.last_error = f"Controle audio impossible: {exc}"
            self._refresh()

    def stop(self) -> None:
        self._refresh()
        process = self.process
        self.process = None
        self.paused = False

        if not process or process.poll() is not None:
            return

        try:
            if hasattr(signal, "SIGCONT"):
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGCONT)
                else:
                    os.kill(process.pid, signal.SIGCONT)
        except OSError:
            pass

        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except OSError:
            pass

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=1.0)

    def status_text(self) -> str:
        self._refresh()
        if self.last_error:
            return self.last_error
        if self.process:
            return "Audio en pause" if self.paused else "Lecture audio en cours"
        return "Lecture audio terminee"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                values[key] = value
        else:
            values.setdefault("RAW_TOKEN", line)

    return values


def env_value(env_file: dict[str, str], key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or env_file.get(key) or default


def int_value(env_file: dict[str, str], key: str, default: int) -> int:
    value = env_value(env_file, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Valeur ignoree pour {key}: {value!r}. Valeur par defaut: {default}.")
        return default


def resolve_app_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return APP_DIR / path


def load_config() -> Config:
    env_file = read_env_file(ENV_PATH)
    token = next((env_value(env_file, key) for key in TOKEN_KEYS if env_value(env_file, key)), None)
    token = token or env_file.get("RAW_TOKEN")

    if not token:
        raise RuntimeError(
            "Token GitHub Models introuvable. Mets GITHUB_MODELS_TOKEN=... dans .env "
            "ou laisse la cle brute seule sur la premiere ligne."
        )

    return Config(
        token=token,
        model=env_value(env_file, "GITHUB_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        endpoint=env_value(env_file, "GITHUB_MODELS_ENDPOINT", DEFAULT_ENDPOINT) or DEFAULT_ENDPOINT,
        api_version=env_value(env_file, "GITHUB_MODELS_API_VERSION", DEFAULT_API_VERSION)
        or DEFAULT_API_VERSION,
        camera_index=int_value(env_file, "CAMERA_INDEX", 0),
        hold_frames=max(1, int_value(env_file, "TOUCH_HOLD_FRAMES", 5)),
        max_image_side=max(320, int_value(env_file, "MAX_IMAGE_SIDE", 1024)),
        jpeg_quality=min(95, max(40, int_value(env_file, "JPEG_QUALITY", 85))),
        hand_model_path=resolve_app_path(
            env_value(env_file, "HAND_LANDMARKER_MODEL_PATH", DEFAULT_HAND_LANDMARKER_MODEL_PATH)
            or DEFAULT_HAND_LANDMARKER_MODEL_PATH
        ),
        hand_model_url=env_value(
            env_file, "HAND_LANDMARKER_MODEL_URL", DEFAULT_HAND_LANDMARKER_MODEL_URL
        )
        or DEFAULT_HAND_LANDMARKER_MODEL_URL,
        tts_voice=env_value(env_file, "TTS_VOICE", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE,
        tts_rate=env_value(env_file, "TTS_RATE", DEFAULT_TTS_RATE) or DEFAULT_TTS_RATE,
    )


def ensure_hand_landmarker_model(config: Config) -> Path:
    model_path = config.hand_model_path
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_name(f"{model_path.name}.tmp")
    print(f"Telechargement du modele MediaPipe: {model_path}")

    try:
        with requests.get(config.hand_model_url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        file.write(chunk)
        temp_path.replace(model_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return model_path


def distance(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def index_touches_middle(
    hand_landmarks: Sequence[object], frame_shape: tuple[int, int, int]
) -> tuple[bool, float, float]:
    height, width = frame_shape[:2]
    landmarks = hand_landmarks

    index_tip = (
        int(landmarks[INDEX_TIP].x * width),
        int(landmarks[INDEX_TIP].y * height),
    )
    middle_tip = (
        int(landmarks[MIDDLE_TIP].x * width),
        int(landmarks[MIDDLE_TIP].y * height),
    )

    xs = [landmark.x * width for landmark in landmarks]
    ys = [landmark.y * height for landmark in landmarks]
    hand_diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    threshold = max(18.0, min(48.0, hand_diagonal * 0.10))
    tip_distance = distance(index_tip, middle_tip)

    return tip_distance <= threshold, tip_distance, threshold


def landmark_to_pixel(landmark, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
    height, width = frame_shape[:2]
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return min(width - 1, max(0, x)), min(height - 1, max(0, y))


def draw_hand_landmarks(frame, hand_landmarks: Sequence[object]) -> None:
    points = [landmark_to_pixel(landmark, frame.shape) for landmark in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 220, 255), 2, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, point, 4, (40, 255, 100), -1, cv2.LINE_AA)


def draw_status(frame, text: str, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def wrap_text(text: str, max_width: int, scale: float = 0.62, thickness: int = 1) -> list[str]:
    lines: list[str] = []
    current = ""

    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
        current = word

    if current:
        lines.append(current)

    return lines


def draw_panel(frame, title: str, body: Iterable[str], footer: str | None = None) -> None:
    height, width = frame.shape[:2]
    panel_width = min(width - 36, 760)
    max_text_width = panel_width - 42
    body_lines: list[str] = []

    for paragraph in body:
        body_lines.extend(wrap_text(paragraph, max_text_width))

    if footer:
        body_lines.append("")
        body_lines.extend(wrap_text(footer, max_text_width, scale=0.56))

    line_height = 28
    panel_height = 82 + line_height * max(1, len(body_lines))
    panel_height = min(panel_height, height - 36)
    x = (width - panel_width) // 2
    y = max(18, (height - panel_height) // 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 1)

    cv2.putText(
        frame,
        title,
        (x + 22, y + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    text_y = y + 82
    for line in body_lines[: max(1, (panel_height - 76) // line_height)]:
        cv2.putText(
            frame,
            line,
            (x + 22, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if line else 0.3,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        text_y += line_height


def encode_frame_as_jpeg_base64(frame, max_side: int, quality: int) -> str:
    height, width = frame.shape[:2]
    longest_side = max(width, height)
    image = frame

    if longest_side > max_side:
        scale = max_side / float(longest_side)
        resized_width = int(width * scale)
        resized_height = int(height * scale)
        image = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Impossible d'encoder l'image en JPEG.")

    return base64.b64encode(buffer.tobytes()).decode("ascii")


def extract_api_error(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        message = data.get("message")
        if message:
            return str(message)

    return str(data)[:500]


def describe_image(frame, config: Config) -> str:
    image_b64 = encode_frame_as_jpeg_base64(frame, config.max_image_side, config.jpeg_quality)
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 350,
        "temperature": 0.2,
    }

    response = requests.post(
        config.endpoint,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": config.api_version,
        },
        json=payload,
        timeout=(10, 90),
    )

    if response.status_code >= 400:
        raise RuntimeError(f"GitHub Models HTTP {response.status_code}: {extract_api_error(response)}")

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Reponse GitHub Models inattendue: {data}") from exc

    if isinstance(content, list):
        content = " ".join(str(part.get("text", part)) for part in content)

    return str(content).strip()


def run() -> int:
    try:
        config = load_config()
        hand_model_path = ensure_hand_landmarker_model(config)
    except RuntimeError as exc:
        print(exc)
        return 1
    except requests.RequestException as exc:
        print(f"Impossible de telecharger le modele MediaPipe: {exc}")
        return 1

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la camera index {config.camera_index}.")
        return 1

    state = "live"
    touch_frames = 0
    frozen_frame = None
    result_text = ""
    last_capture_at = 0.0
    last_timestamp_ms = 0
    speech = SpeechPlayer(config.tts_voice, config.tts_rate)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    with hand_landmarker.HandLandmarker.create_from_options(options) as hands:
        while True:
            if state == "live":
                ok, frame = cap.read()
                if not ok:
                    print("Impossible de lire une image depuis la camera.")
                    break

                display = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
                last_timestamp_ms = timestamp_ms
                results = hands.detect_for_video(mp_image, timestamp_ms)

                is_touching = False
                if results.hand_landmarks:
                    hand_landmarks = results.hand_landmarks[0]
                    is_touching, tip_distance, threshold = index_touches_middle(
                        hand_landmarks, frame.shape
                    )
                    draw_hand_landmarks(display, hand_landmarks)

                    if is_touching:
                        touch_frames += 1
                        draw_status(
                            display,
                            f"Geste detecte {touch_frames}/{config.hold_frames} - distance {tip_distance:.0f}px",
                            (80, 255, 120),
                        )
                    else:
                        touch_frames = 0
                        draw_status(
                            display,
                            f"Live - touche index + majeur pour capturer | seuil {threshold:.0f}px | Q quitter",
                        )
                else:
                    touch_frames = 0
                    draw_status(display, "Live - montre ta main, puis touche index + majeur | Q quitter")

                if touch_frames >= config.hold_frames and time.time() - last_capture_at > 1.0:
                    frozen_frame = frame.copy()
                    last_capture_at = time.time()
                    touch_frames = 0
                    state = "confirm"
                    continue

            elif state == "confirm" and frozen_frame is not None:
                display = frozen_frame.copy()
                draw_panel(
                    display,
                    "Voulez-vous envoyer ?",
                    ["La camera est gelee sur cette image."],
                    "Oui: O / Y / Entree    Non: N / Echap    Quitter: Q",
                )

            elif state == "sending" and frozen_frame is not None:
                display = frozen_frame.copy()
                draw_panel(display, "Envoi a l'IA...", ["Patientez pendant l'analyse de l'image."])
                cv2.imshow(WINDOW_TITLE, display)
                cv2.waitKey(1)
                try:
                    result_text = describe_image(frozen_frame, config)
                    print("\nDescription IA:\n" + result_text + "\n")
                    speech.speak(result_text)
                    state = "result"
                except Exception as exc:  # noqa: BLE001 - affiche l'erreur utile a l'utilisateur.
                    speech.stop()
                    result_text = str(exc)
                    print("\nErreur IA:\n" + result_text + "\n")
                    state = "error"
                continue

            elif state == "result" and frozen_frame is not None:
                display = frozen_frame.copy()
                draw_panel(
                    display,
                    "Description IA",
                    [result_text or "Aucune description recue."],
                    f"{speech.status_text()}    Pause/reprise: P    Continuer: C / Entree    Quitter: Q",
                )

            elif state == "error" and frozen_frame is not None:
                display = frozen_frame.copy()
                draw_panel(
                    display,
                    "Erreur IA",
                    [result_text or "Erreur inconnue."],
                    "Continuer: C / Entree    Quitter: Q",
                )

            else:
                state = "live"
                continue

            cv2.imshow(WINDOW_TITLE, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                speech.stop()
                break

            if state == "confirm":
                if key in (ord("o"), ord("O"), ord("y"), ord("Y"), 13, 10):
                    state = "sending"
                elif key in (ord("n"), ord("N"), 27):
                    frozen_frame = None
                    state = "live"

            elif state == "result":
                if key in (ord("p"), ord("P")):
                    speech.toggle_pause()
                elif key in (ord("c"), ord("C"), 13, 10, 27):
                    speech.stop()
                    frozen_frame = None
                    result_text = ""
                    state = "live"

            elif state == "error" and key in (ord("c"), ord("C"), 13, 10, 27):
                speech.stop()
                frozen_frame = None
                result_text = ""
                state = "live"

    speech.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
