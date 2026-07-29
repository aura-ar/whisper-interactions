import dataclasses
import logging
import os
import time
import torch
import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.agents.utils import AudioBuffer
import threading

logger = logging.getLogger(__name__)


@dataclass
class WhisperOptions:
    language: str
    model: str
    device: str
    compute_type: str
    model_cache_directory: Optional[str]

_SHARED_MODEL = None
_MODEL_LOCK = threading.Lock()
class WhisperSTT(stt.STT):
    def __init__(
        self,
        model: str = "small",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        model_cache_directory: Optional[str] = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._opts = WhisperOptions(
            language=language,
            model=model,
            device=device,
            compute_type=compute_type,
            model_cache_directory=model_cache_directory,
        )

        self._model = None
        self._initialize_model()

    def _initialize_model(self):

        global _SHARED_MODEL

        if _SHARED_MODEL is not None:
            self._model = _SHARED_MODEL
            logger.info("Using cached Whisper model.")
            return


        with _MODEL_LOCK:

            if _SHARED_MODEL is not None:
                self._model = _SHARED_MODEL
                logger.info("Using cached Whisper model.")
                return


            logger.info(
                "Loading Whisper model: %s on %s (%s)",
                self._opts.model,
                self._opts.device,
                self._opts.compute_type,
            )


            self._model = WhisperModel(
                model_size_or_path=self._opts.model,
                device=self._opts.device,
                compute_type=self._opts.compute_type,
            )


            _SHARED_MODEL = self._model


            logger.info(
                "Whisper model loaded successfully."
            )
        

    def _sanitize_options(self, language: Optional[str] = None) -> WhisperOptions:
        options = dataclasses.replace(self._opts)
        if language:
            options.language = language
        return options

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            options = self._sanitize_options(language=language)

            logger.info("Received audio buffer. Transcribing...")

            logger.info("Whisper received audio")
            audio_frame = rtc.combine_audio_frames(buffer)
            wav_bytes = audio_frame.to_wav_bytes()

            audio_int16 = np.frombuffer(wav_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            start_time = time.time()

            segments, info = self._model.transcribe(
                audio_float32,
                language=options.language,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.55,
            )

            segments_list = list(segments)
            full_text = " ".join(
                segment.text.strip()
                for segment in segments_list
                if segment.text.strip()
            ).strip()

            elapsed = time.time() - start_time
            logger.info("Whisper STT completed in %.2fs. Text: %s", elapsed, full_text)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text,
                        language=options.language,
                    )
                ],
            )

        except Exception as e:
            logger.exception("Whisper STT error: %s", e)
            raise APIConnectionError() from e
    def __del__(self):
        global _SHARED_MODEL

        gc.collect()

        try:
            torch.cuda.empty_cache()
        except:
            pass