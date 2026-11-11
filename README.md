# Local LiveKit Voice Assistant Demo

This project demonstrates a local real-time voice assistant using LiveKit.

## Features

* LiveKit local server for real-time audio streaming
* Browser microphone input through LiveKit Agents Playground
* Python LiveKit agent
* Local Whisper STT using faster-whisper
* Local LLM response generation using LMStudio
* Optional TTS support using external TTS providers

## Pipeline

```text
User Speech
→ LiveKit Audio Stream
→ Whisper Speech-to-Text
→ LMStudio Local LLM
→ AI Text Response
```

## Experimental CCB

The project implements a factorized contextual combinatorial Thompson sampler
in `contextual_bandit.py`. The agent first asks for the participant's initial
sandwich choice. Before each follow-up nudge it selects one nudge and one
sandwich goal as a joint action. Its posterior has separate context-dependent
effects for each nudge, each goal, and their pair; the observed joint reward
updates all three blocks. A switch to the selected goal is rewarded and ends
the sequence immediately. Previously used nudges and goals, plus the
participant's current choice, are excluded from later selections in that
interaction.

The sample action groups are Social Influence, Warning, and Anchoring; and
BLT, Chicken, and Prawn. The provisional reward is `+1.0` when the participant
switches to the selected goal after the first nudge, decreasing by `0.25` for
each later successful attempt. An unsuccessful nudge receives `-0.25`; a
choice that already matches the goal receives `0`. The learned posterior is
stored locally in `.ccb_state.json`.

This is a prototype: it recognizes sandwich choices by matching those sample
names in the transcript and enters the flow after the agent asks about weekend
plans. Replace the sample actions, context features, and transition detection
when the menu and conversation design are finalized. Nudge wording is
constrained not to invent social proof, warnings, or product claims.

## Technologies Used

* Python
* LiveKit Agents
* LiveKit Agents Playground
* faster-whisper
* Ollama
* Silero VAD

## How to Run

### 1. Start LiveKit Server

```powershell
cd D:\livekit
.\livekit-server.exe --dev
```

### 2. Start LMStudio

Start LMStudio and enable the OpenAI-compatible API endpoint on your local machine.

Typical LMStudio API endpoint:

```text
http://localhost:8080/api/v1
```

If LMStudio is already running, make sure the host and port are not in use by another service.

### 3. Start LiveKit Agent

```powershell
cd D:\livekit
.\livekit_env\Scripts\activate
python livekit_agent.py dev
```

### 4. Start LiveKit Playground

```powershell
cd D:\agents-playground
npm run dev
```

Open:

```text
http://localhost:3000
```

Click **Connect**, allow microphone permission, and speak.

## Demo Output

The system receives microphone audio from the browser, transcribes it using Whisper, and sends the transcript to the local LMStudio LLM for response generation.

## Note

The `.env` file is not uploaded to GitHub because it may contain private API keys.
