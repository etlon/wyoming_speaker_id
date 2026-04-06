"""Speaker profile database for enrollment and identification.

Folder-based structure:
  profiles_dir/
    cedric/
      sample1.webm
      sample2.mp3
      sample3.wav
      .embedding.json   (cached embedding, auto-regenerated)
    lovis/
      sample1.webm
      .embedding.json

Each subfolder is a speaker. Audio files are the voice samples.
The .embedding.json cache is recomputed when samples change.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a", ".opus"}
EMBEDDING_CACHE = ".embedding.json"

# Lazy-load resemblyzer to avoid slow import at module level
_encoder = None


def _get_encoder():
    """Lazy-load the speaker encoder model."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder("cpu")
        _LOGGER.info("Speaker encoder model loaded")
    return _encoder


def _audio_to_numpy(file_path: Path) -> Optional[np.ndarray]:
    """Convert any audio file to 16kHz mono float32 numpy array via ffmpeg."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(file_path),
             "-ar", "16000", "-ac", "1", "-f", "s16le",
             "-acodec", "pcm_s16le", "pipe:1"],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            _LOGGER.error("ffmpeg failed for %s: %s", file_path.name,
                          result.stderr.decode(errors="replace")[-300:])
            return None
        raw = np.frombuffer(result.stdout, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0
    except Exception as e:
        _LOGGER.error("Audio conversion failed for %s: %s", file_path.name, e)
        return None


class SpeakerProfile:
    """A single speaker's voice profile."""

    def __init__(self, name: str, embedding: np.ndarray):
        self.name = name
        self.embedding = embedding


class SpeakerDatabase:
    """Manages speaker profiles using a folder-based structure."""

    def __init__(
        self,
        profiles_dir: str = "/share/wyoming-speaker-id/profiles",
        similarity_threshold: float = 0.75,
        unknown_label: str = "Unbekannt",
    ):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.unknown_label = unknown_label
        self.profiles: dict[str, SpeakerProfile] = {}
        self.load_profiles()

    def _get_sample_files(self, speaker_dir: Path) -> list[Path]:
        """Get all audio sample files in a speaker directory."""
        files = []
        for f in sorted(speaker_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(f)
        return files

    def _samples_hash(self, sample_files: list[Path]) -> str:
        """Compute a hash of sample filenames + modification times."""
        h = hashlib.md5()
        for f in sample_files:
            h.update(f"{f.name}:{f.stat().st_mtime_ns}".encode())
        return h.hexdigest()

    def _load_cached_embedding(self, speaker_dir: Path, samples_hash: str) -> Optional[np.ndarray]:
        """Load cached embedding if it matches the current samples."""
        cache_file = speaker_dir / EMBEDDING_CACHE
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            if data.get("samples_hash") == samples_hash:
                return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            pass
        return None

    def _save_cached_embedding(self, speaker_dir: Path, embedding: np.ndarray, samples_hash: str):
        """Save embedding cache."""
        cache_file = speaker_dir / EMBEDDING_CACHE
        with open(cache_file, "w") as f:
            json.dump({
                "samples_hash": samples_hash,
                "embedding": embedding.tolist(),
            }, f)

    def _compute_embedding(self, sample_files: list[Path]) -> Optional[np.ndarray]:
        """Compute averaged embedding from audio files."""
        encoder = _get_encoder()
        from resemblyzer import preprocess_wav

        embeddings = []
        for audio_file in sample_files:
            audio = _audio_to_numpy(audio_file)
            if audio is None:
                continue
            processed = preprocess_wav(audio, source_sr=16000)
            if len(processed) < 1600:  # < 0.1s
                _LOGGER.warning("Sample too short, skipping: %s", audio_file.name)
                continue
            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)

        if not embeddings:
            return None

        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)
        return avg

    def load_profiles(self):
        """Scan profiles directory and load/recompute all speaker embeddings."""
        self.profiles.clear()
        for speaker_dir in sorted(self.profiles_dir.iterdir()):
            if not speaker_dir.is_dir() or speaker_dir.name.startswith(".") or speaker_dir.name.startswith("_"):
                continue

            name = speaker_dir.name
            samples = self._get_sample_files(speaker_dir)
            if not samples:
                _LOGGER.debug("No audio samples for '%s', skipping", name)
                continue

            s_hash = self._samples_hash(samples)
            embedding = self._load_cached_embedding(speaker_dir, s_hash)

            if embedding is None:
                _LOGGER.info("Computing embedding for '%s' (%d samples)...", name, len(samples))
                embedding = self._compute_embedding(samples)
                if embedding is None:
                    _LOGGER.error("Failed to compute embedding for '%s'", name)
                    continue
                self._save_cached_embedding(speaker_dir, embedding, s_hash)
            else:
                _LOGGER.debug("Using cached embedding for '%s'", name)

            self.profiles[name] = SpeakerProfile(name=name, embedding=embedding)

        _LOGGER.info("Loaded %d speaker profiles", len(self.profiles))

    def save_sample(self, speaker_name: str, audio_data: bytes, filename: str) -> Path:
        """Save an audio sample to a speaker's folder."""
        safe_name = "".join(c for c in speaker_name if c.isalnum() or c in "-_ ").strip()
        if not safe_name:
            raise ValueError("Invalid speaker name")
        speaker_dir = self.profiles_dir / safe_name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        filepath = speaker_dir / filename
        with open(filepath, "wb") as f:
            f.write(audio_data)
        _LOGGER.info("Saved sample: %s/%s", safe_name, filename)
        return filepath

    def delete_sample(self, speaker_name: str, filename: str) -> bool:
        """Delete a single audio sample."""
        filepath = self.profiles_dir / speaker_name / filename
        if not filepath.exists() or filepath.suffix.lower() not in AUDIO_EXTENSIONS:
            return False
        filepath.unlink()
        # Remove embedding cache so it gets recomputed
        cache = self.profiles_dir / speaker_name / EMBEDDING_CACHE
        if cache.exists():
            cache.unlink()
        _LOGGER.info("Deleted sample: %s/%s", speaker_name, filename)
        return True

    def move_sample(self, from_speaker: str, filename: str, to_speaker: str) -> bool:
        """Move a sample from one speaker to another."""
        import shutil
        src = self.profiles_dir / from_speaker / filename
        if not src.exists() or src.suffix.lower() not in AUDIO_EXTENSIONS:
            return False
        dest_dir = self.profiles_dir / to_speaker
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        shutil.move(str(src), str(dest))
        # Invalidate caches
        for d in [self.profiles_dir / from_speaker, dest_dir]:
            cache = d / EMBEDDING_CACHE
            if cache.exists():
                cache.unlink()
        _LOGGER.info("Moved sample: %s/%s -> %s/%s", from_speaker, filename, to_speaker, filename)
        return True

    def recompute_speaker(self, speaker_name: str) -> bool:
        """Recompute embedding for a speaker from their current samples."""
        speaker_dir = self.profiles_dir / speaker_name
        if not speaker_dir.is_dir():
            return False
        samples = self._get_sample_files(speaker_dir)
        if not samples:
            # No samples left — remove profile
            if speaker_name in self.profiles:
                del self.profiles[speaker_name]
            cache = speaker_dir / EMBEDDING_CACHE
            if cache.exists():
                cache.unlink()
            return True

        embedding = self._compute_embedding(samples)
        if embedding is None:
            return False

        s_hash = self._samples_hash(samples)
        self._save_cached_embedding(speaker_dir, embedding, s_hash)
        self.profiles[speaker_name] = SpeakerProfile(name=speaker_name, embedding=embedding)
        _LOGGER.info("Recomputed embedding for '%s' (%d samples)", speaker_name, len(samples))
        return True

    def delete_speaker(self, speaker_name: str) -> bool:
        """Delete a speaker and all their samples."""
        import shutil
        speaker_dir = self.profiles_dir / speaker_name
        if not speaker_dir.is_dir():
            return False
        shutil.rmtree(speaker_dir)
        self.profiles.pop(speaker_name, None)
        _LOGGER.info("Deleted speaker: %s", speaker_name)
        return True

    def list_speakers(self) -> list[dict]:
        """List all speakers (and _unknown) with their samples."""
        result = []
        for speaker_dir in sorted(self.profiles_dir.iterdir()):
            if not speaker_dir.is_dir() or speaker_dir.name.startswith("."):
                continue
            samples = self._get_sample_files(speaker_dir)
            result.append({
                "name": speaker_dir.name,
                "samples": [f.name for f in samples],
                "enrolled": speaker_dir.name in self.profiles,
            })
        return result

    def identify_speaker(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[Optional[str], Optional[str], float]:
        """Identify the speaker from an audio sample."""
        if not self.profiles:
            return self.unknown_label, None, 0.0

        encoder = _get_encoder()
        from resemblyzer import preprocess_wav

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        processed = preprocess_wav(audio, source_sr=sample_rate)
        if len(processed) < 8000:
            _LOGGER.warning("Audio too short for speaker ID (%.2fs)", len(processed) / 16000)
            return self.unknown_label, None, 0.0

        query_embedding = encoder.embed_utterance(processed)

        best_name = self.unknown_label
        best_similarity = 0.0

        for profile in self.profiles.values():
            similarity = float(np.dot(query_embedding, profile.embedding))
            _LOGGER.debug("Similarity with '%s': %.3f", profile.name, similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                if similarity >= self.similarity_threshold:
                    best_name = profile.name

        _LOGGER.info("Speaker: %s (%.3f)", best_name, best_similarity)
        return best_name, best_name if best_name != self.unknown_label else None, best_similarity
