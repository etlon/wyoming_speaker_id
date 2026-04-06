"""Speaker profile database for enrollment and identification."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

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


class SpeakerProfile:
    """A single speaker's voice profile."""

    def __init__(self, name: str, user_id: str, embedding: np.ndarray):
        self.name = name
        self.user_id = user_id
        self.embedding = embedding

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "user_id": self.user_id,
            "embedding": self.embedding.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        return cls(
            name=data["name"],
            user_id=data["user_id"],
            embedding=np.array(data["embedding"], dtype=np.float32),
        )


class SpeakerDatabase:
    """Manages speaker profiles and performs identification."""

    def __init__(
        self,
        profiles_dir: str = "/data/profiles",
        similarity_threshold: float = 0.75,
        unknown_label: str = "Unbekannt",
    ):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.unknown_label = unknown_label
        self.profiles: dict[str, SpeakerProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load all speaker profiles from disk."""
        self.profiles.clear()
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, "r") as f:
                    data = json.load(f)
                profile = SpeakerProfile.from_dict(data)
                self.profiles[profile.user_id] = profile
                _LOGGER.info("Loaded speaker profile: %s (%s)", profile.name, profile.user_id)
            except Exception as e:
                _LOGGER.error("Failed to load profile %s: %s", profile_file, e)

        _LOGGER.info("Loaded %d speaker profiles", len(self.profiles))

    def _save_profile(self, profile: SpeakerProfile):
        """Save a speaker profile to disk."""
        profile_path = self.profiles_dir / f"{profile.user_id}.json"
        with open(profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        _LOGGER.info("Saved profile: %s -> %s", profile.name, profile_path)

    def enroll_speaker(
        self, name: str, user_id: str, audio_samples: list[np.ndarray]
    ) -> SpeakerProfile:
        """Enroll a new speaker from one or more audio samples.

        Args:
            name: Display name of the speaker
            user_id: Home Assistant user ID or unique identifier
            audio_samples: List of audio arrays (16kHz, float32, mono)

        Returns:
            The created SpeakerProfile
        """
        encoder = _get_encoder()
        from resemblyzer import preprocess_wav

        embeddings = []
        for audio in audio_samples:
            # Ensure float32 in range [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            processed = preprocess_wav(audio, source_sr=16000)
            if len(processed) < 1600:  # Less than 0.1s
                _LOGGER.warning("Audio sample too short, skipping")
                continue

            emb = encoder.embed_utterance(processed)
            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No valid audio samples provided for enrollment")

        # Average all embeddings for a more robust profile
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        profile = SpeakerProfile(
            name=name, user_id=user_id, embedding=avg_embedding
        )
        self.profiles[user_id] = profile
        self._save_profile(profile)
        _LOGGER.info(
            "Enrolled speaker '%s' with %d audio samples", name, len(embeddings)
        )
        return profile

    def identify_speaker(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[Optional[str], Optional[str], float]:
        """Identify the speaker from an audio sample.

        Args:
            audio: Audio array (mono)
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (speaker_name, user_id, confidence).
            If no match, returns (unknown_label, None, 0.0)
        """
        if not self.profiles:
            _LOGGER.debug("No speaker profiles enrolled")
            return self.unknown_label, None, 0.0

        encoder = _get_encoder()
        from resemblyzer import preprocess_wav

        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        processed = preprocess_wav(audio, source_sr=sample_rate)

        if len(processed) < 8000:  # Less than 0.5s of audio
            _LOGGER.warning("Audio too short for reliable speaker ID (%.2fs)", len(processed) / sample_rate)
            return self.unknown_label, None, 0.0

        query_embedding = encoder.embed_utterance(processed)

        best_name = self.unknown_label
        best_user_id = None
        best_similarity = 0.0

        for profile in self.profiles.values():
            similarity = float(
                np.dot(query_embedding, profile.embedding)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(profile.embedding))
            )
            _LOGGER.debug(
                "Similarity with '%s': %.3f (threshold: %.3f)",
                profile.name, similarity, self.similarity_threshold,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                if similarity >= self.similarity_threshold:
                    best_name = profile.name
                    best_user_id = profile.user_id

        _LOGGER.info(
            "Speaker identified: %s (confidence: %.3f)", best_name, best_similarity
        )
        return best_name, best_user_id, best_similarity

    def delete_speaker(self, user_id: str) -> bool:
        """Delete a speaker profile."""
        if user_id not in self.profiles:
            return False
        profile_path = self.profiles_dir / f"{user_id}.json"
        if profile_path.exists():
            profile_path.unlink()
        del self.profiles[user_id]
        _LOGGER.info("Deleted speaker profile: %s", user_id)
        return True

    def list_speakers(self) -> list[dict]:
        """List all enrolled speakers."""
        return [
            {"name": p.name, "user_id": p.user_id}
            for p in self.profiles.values()
        ]
