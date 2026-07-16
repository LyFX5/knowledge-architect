import logging
from pathlib import Path

from pydub import AudioSegment

# import whisper # moved from openai whisper to faster-whisper
from faster_whisper import WhisperModel

from app.config.settings import settings
from app.transcription.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperService:
    """Service for transcribing audio using Whisper."""

    def __init__(self):
        # self.model = whisper.load_model(settings.WHISPER_MODEL)
        self.model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info(
            f"Whisper model '{settings.WHISPER_MODEL_SIZE}' loaded successfully"
        )

    def split_audio_into_chunks(
        audio_path: str,
        chunk_length_ms: int = 5 * 60 * 1000,  # 5 minutes
    ) -> list[str]:
        """
        Split an audio file into fixed-size chunks.

        Returns a list of temporary chunk filenames.
        """
        audio = AudioSegment.from_file(audio_path)
        audio_path = Path(audio_path)
        chunk_paths = []

        for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
            chunk = audio[start : start + chunk_length_ms]
            chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(str(chunk_path))

        return chunk_paths

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file

        Returns:
            TranscriptionResult with text and paths
        """
        logger.info(f"Transcribing audio: {audio_path}")

        audio = AudioSegment.from_file(audio_path)
        duration_minutes = len(audio) / 1000 / 60

        if duration_minutes <= 12:
            # result = self.model.transcribe(audio_path)
            # text = result["text"].strip()
            segments, info = self.model.transcribe(audio_path)
            text = "".join(segment.text for segment in segments).strip()

        else:
            chunk_paths = self.split_audio_into_chunks(audio_path)
            texts = []

            for chunk_path in chunk_paths:
                segments, info = self.model.transcribe(chunk_path)
                text = "".join(segment.text for segment in segments).strip()
                texts.append(text)

            # cleanup
            for chunk_path in chunk_paths:
                try:
                    Path(chunk_path).unlink()
                except Exception:
                    pass

            text = "\n\n".join(texts)

        # Generate transcript filename from audio filename
        audio_file = Path(audio_path)
        transcript_filename = f"{audio_file.stem}.txt"
        transcript_path = settings.TRANSCRIPTS_DIR / transcript_filename

        # Save transcript to file
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Transcription saved to: {transcript_path}")

        return TranscriptionResult(
            text=text,
            audio_path=str(audio_path),
            transcript_path=str(transcript_path),
            language=info.language,
        )
