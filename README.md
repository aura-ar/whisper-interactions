# Local LiveKit Voice Assistant Demo

This project demonstrates a local real-time voice assistant using LiveKit.

## Features

* LiveKit local server for real-time audio streaming
* Browser microphone input through LiveKit Agents Playground
* Python LiveKit agent
* Local Whisper STT using faster-whisper
* Local LLM response generation using Ollama
* Optional TTS support using external TTS providers

## Pipeline

```text
User Speech
→ LiveKit Audio Stream
→ Whisper Speech-to-Text
→ Ollama Local LLM
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

### 2. Start Ollama

```powershell
ollama serve
```

If Ollama is already running, this may show `address already in use`.

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

The system receives microphone audio from the browser, transcribes it using Whisper, and sends the transcript to the local Ollama LLM for response generation.

## Note

The `.env` file is not uploaded to GitHub because it may contain private API keys.
