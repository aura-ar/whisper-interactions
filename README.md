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
