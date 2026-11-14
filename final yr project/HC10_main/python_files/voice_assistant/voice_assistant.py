#!/usr/bin/env python3
"""
voice_commander.py
──────────────────
Records a voice command, transcribes with Whisper (offline),
extracts color + shape via keyword matching, then launches
voice_assistant as a ROS2 node with those two parameters.

Dependencies (install once):
  pip install openai-whisper sounddevice scipy numpy

Usage:
  python3 voice_commander.py
  python3 voice_commander.py --duration 5        # longer recording
  python3 voice_commander.py --device 2          # specific mic index
  python3 voice_commander.py --model medium      # larger Whisper model
  python3 voice_commander.py --list-devices      # show mic list
"""

import argparse
import subprocess
import sys
import tempfile
import os
import re

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--duration",     type=float, default=4.0,
                    help="Recording length in seconds (default 4)")
parser.add_argument("--device",       type=int,   default=None,
                    help="Input device index (default: system default)")
parser.add_argument("--model",        type=str,   default="base",
                    choices=["tiny","base","small","medium","large"],
                    help="Whisper model size (default: base)")
parser.add_argument("--samplerate",   type=int,   default=16000,
                    help="Recording sample rate (default 16000)")
parser.add_argument("--list-devices", action="store_true",
                    help="List audio input devices and exit")
parser.add_argument("--package",      type=str,   default="hello_moveit",
                    help="ROS2 package name (default: hello_moveit)")
parser.add_argument("--executable",   type=str,   default="voice_assistant",
                    help="ROS2 executable name (default: voice_assistant)")
args = parser.parse_args()

# ─── Optional: list devices ───────────────────────────────────────────────────
import sounddevice as sd

if args.list_devices:
    print(sd.query_devices())
    sys.exit(0)

# ─── Color keyword map → canonical name ───────────────────────────────────────
# Each list contains synonyms / alternate phrasings recognised in speech.
COLOR_KEYWORDS: dict[str, list[str]] = {
    "red":    ["red", "crimson", "scarlet", "maroon", "pink"],
    "green":  ["green", "lime", "olive", "teal", "emerald"],
    "blue":   ["blue", "navy", "cobalt", "azure", "indigo", "cyan"],
    "yellow": ["yellow", "amber", "gold", "golden"],
    "orange": ["orange"],
    "purple": ["purple", "violet", "magenta", "lavender"],
    "white":  ["white", "ivory", "cream"],
    "black":  ["black", "dark", "ebony"],
    "brown":  ["brown", "beige", "tan", "khaki"],
}

# ─── Shape keyword map ─────────────────────────────────────────────────────────
SHAPE_KEYWORDS: dict[str, list[str]] = {
    "cylinder": ["cylinder", "cylindrical", "tube", "can", "round", "circular",
                 "circle", "pipe", "rod", "drum"],
    "cuboid":   ["cuboid", "box", "cube", "block", "rectangular", "rectangle",
                 "square", "brick", "carton", "package", "prism"],
}

# ─── Extraction helpers ────────────────────────────────────────────────────────
def extract_color(text: str) -> str | None:
    text_lower = text.lower()
    for canonical, synonyms in COLOR_KEYWORDS.items():
        for syn in synonyms:
            # whole-word match
            if re.search(r'\b' + re.escape(syn) + r'\b', text_lower):
                return canonical
    return None


def extract_shape(text: str) -> str | None:
    text_lower = text.lower()
    for canonical, synonyms in SHAPE_KEYWORDS.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', text_lower):
                return canonical
    return None


# ─── Recording ────────────────────────────────────────────────────────────────
def record_audio(duration: float, samplerate: int, device) -> "np.ndarray":
    import numpy as np
    print(f"\n🎙  Recording for {duration}s …  (speak now)")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("   Recording done.")
    return audio.flatten()


# ─── Transcription ────────────────────────────────────────────────────────────
def transcribe(audio: "np.ndarray", samplerate: int, model_name: str) -> str:
    import numpy as np
    import scipy.io.wavfile as wav
    import whisper

    # Save to temp wav so Whisper can read it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        wav.write(tmp_path, samplerate, (audio * 32767).astype(np.int16))

    try:
        print(f"   Loading Whisper '{model_name}' model …")
        model = whisper.load_model(model_name, device="cpu")
        result = model.transcribe(tmp_path, language="en", fp16=False)
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


# ─── Launch voice_assistant ───────────────────────────────────────────────────
def launch_node(color: str, shape: str) -> None:
    cmd = [
        "ros2", "run",
        args.package,
        args.executable,
        "--ros-args",
        "-p", f"color:={color}",
        "-p", f"shape:={shape}",
    ]
    print(f"\n🚀 Launching: {' '.join(cmd)}")
    # Replace current process so Ctrl-C reaches the node directly
    os.execvp("ros2", cmd)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Record
    audio = record_audio(args.duration, args.samplerate, args.device)

    # 2. Transcribe
    text = transcribe(audio, args.samplerate, args.model)
    print(f"\n📝 Transcribed: \"{text}\"")

    # 3. Extract
    color = extract_color(text)
    shape = extract_shape(text)

    print(f"   Color  : {color or '(not recognised)'}")
    print(f"   Shape  : {shape or '(not recognised)'}")

    # 4. Validate
    if color is None:
        print("\n❌ No colour recognised in the command.")
        print("   Recognised colours:", ", ".join(COLOR_KEYWORDS.keys()))
        print("   Please try again.")
        sys.exit(1)

    if shape is None:
        print("\n❌ No shape recognised in the command.")
        print("   Recognised shapes:", ", ".join(SHAPE_KEYWORDS.keys()))
        print("   Please try again.")
        sys.exit(1)

    print(f"\n✔  Command understood: pick the {color} {shape}")

    # 5. Launch (replaces this process)
    launch_node(color, shape)


if __name__ == "__main__":
    main()