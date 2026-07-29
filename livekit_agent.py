import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    room_io,
)

from livekit.plugins import silero
from livekit.plugins.openai import LLM

from whisper_plugin import WhisperSTT
from livekit.agents import AutoSubscribe

load_dotenv()

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"


class TherapyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a calm, supportive AI therapy assistant. "
                "Reply in 1 line, warm, natural, and emotionally supportive sentence."
            )
        )

async def entrypoint(ctx: JobContext):

    await ctx.connect(
        auto_subscribe=AutoSubscribe.AUDIO_ONLY
    )

    participant = await ctx.wait_for_participant()
    print(f"Participant joined: {participant.identity}")

    def on_track_subscribed(track, publication, participant):
        print(f"TRACK SUBSCRIBED: kind={track.kind} from={participant.identity}")

    ctx.room.on("track_subscribed", on_track_subscribed)

    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.4,
        ),

        stt=WhisperSTT(
            model="/home/nivisha-vivek/models/large-v3-turbo",
            language="en",
            device="cuda",
            compute_type="int8_float16",
        ),

        llm=LLM(
            model=OLLAMA_MODEL,
            api_key="ollama",
            base_url=OLLAMA_URL,
        ),
    )

    await session.start(
        room=ctx.room,
        agent=TherapyVoiceAgent(),
        room_options=room_io.RoomOptions(
            audio_input=True,
            audio_output=False,
            text_output=True,
        ),
    )

    print("Therapy AI Agent Started - LiveKit + Whisper STT + Ollama")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=2000,
            num_idle_processes=1,
        )
    )