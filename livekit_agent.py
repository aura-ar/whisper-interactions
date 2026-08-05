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

LMSTUDIO_URL = "http://127.0.0.1:1234/api/v1"
LMSTUDIO_MODEL = "google/gemma-4-e4b@q4_k_m"
LMSTUDIO_API_KEY = None
WHISPER_MODEL_PATH = "/home/aurora/Documents/Reinforcement/my-venv/lib/python3.12/site-packages"


class TherapyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "Start every sentance with: woof "
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
            model=WHISPER_MODEL_PATH,
            language="en",
            device="cuda",
            compute_type="int8_float16",
        ),

        llm=LLM(
            model=LMSTUDIO_MODEL,
            api_key=LMSTUDIO_API_KEY or None,
            base_url=LMSTUDIO_URL,
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

    print("Therapy AI Agent Started - LiveKit + Whisper STT + LMStudio")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=2000,
            num_idle_processes=1,
        )
    )