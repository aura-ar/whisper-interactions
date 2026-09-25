from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    room_io,
)

import os

import httpx
import openai
from livekit.plugins import silero
from livekit.plugins.openai import LLM

# lms server start and lk agent dev --dev livekit_agent.py

from whisper_plugin import WhisperSTT
from livekit.agents import AutoSubscribe
from dotenv import load_dotenv
from livekit.agents import AgentServer

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL = "google/gemma-4-e4b"
WHISPER_MODEL_PATH = "medium.en"

# default LiveKit websocket URL for local dev when LIVEKIT_URL not set
# os.environ.setdefault("LIVEKIT_URL", "ws://127.0.0.1:7880")

openai_client = openai.AsyncClient(
    api_key="lm-studio",
    base_url=LMSTUDIO_URL,
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, read=15.0)
    ),
)

llm_client = LLM(
    client=openai_client,
    model=LMSTUDIO_MODEL,
)


class SustainabilityVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "Start every sentance with: woof "
            )
        )


load_dotenv(override=True)

print("LIVEKIT_URL =", repr(os.environ.get("LIVEKIT_URL")))
print("LIVEKIT_API_KEY =", repr(os.environ.get("LIVEKIT_API_KEY")))
print("LIVEKIT_API_SECRET set =", bool(os.environ.get("LIVEKIT_API_SECRET")))

server = AgentServer(
    ws_url=os.environ.get("LIVEKIT_URL"),
    api_key=os.environ.get("LIVEKIT_API_KEY"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET"),
)


@server.rtc_session()
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
            model=WHISPER_MODEL_PATH,
            language="en",
            device="cuda",
            compute_type="int8",
        ),

        llm=llm_client,
    )

    await session.start(
        room=ctx.room,
        agent=SustainabilityVoiceAgent(),
        room_options=room_io.RoomOptions(
            audio_input=True,
            audio_output=False,
            text_output=True,
        ),
    )

    print("AI Agent Started - LiveKit + Whisper STT + LMStudio")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=2000,
            num_idle_processes=1,
        )
    )
