/*
 * Frontend demo client for /ws/voice.
 *
 * LIMITATION: microphone capture here records a real WebM/Opus clip via
 * MediaRecorder and sends it as base64 in an "audio" event, but the
 * server-side mock STT does not decode real audio (see
 * app/services/stt.py) -- it will show "[unrecognized audio - STT
 * running in offline mock mode]" unless a real Whisper/faster-whisper
 * model is configured. Text input always works end-to-end.
 */

const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsProtocol}://${location.host}/ws/voice`);

const conversation = document.getElementById("conversation");
const latencyBox = document.getElementById("latencyBox");
const handoffBox = document.getElementById("handoffBox");
const player = document.getElementById("player");
const micBtn = document.getElementById("micBtn");
const recordingIndicator = document.getElementById("recordingIndicator");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  conversation.appendChild(div);
  conversation.scrollTop = conversation.scrollHeight;
}

ws.addEventListener("open", () => addMessage("Connecting…", "system"));

ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "ready":
      addMessage(`Session ready (${msg.session_id.slice(0, 8)})`, "system");
      break;
    case "transcript":
      addMessage(msg.payload.text, "user");
      if (msg.payload.is_mock_stt) addMessage("(offline mock STT — see README limitations)", "system");
      break;
    case "agent_response":
      addMessage(msg.payload.text, "agent");
      if (msg.payload.barge_in_detected) addMessage("(barge-in detected — previous playback interrupted)", "system");
      break;
    case "audio_response": {
      const audioBytes = Uint8Array.from(atob(msg.payload.audio_base64), (c) => c.charCodeAt(0));
      const blob = new Blob([audioBytes], { type: "audio/wav" });
      player.src = URL.createObjectURL(blob);
      if (msg.payload.is_mock_tts) addMessage("(offline mock TTS — silent placeholder audio, see README)", "system");
      break;
    }
    case "metrics":
      latencyBox.textContent = JSON.stringify(msg.payload, null, 2);
      break;
    case "handoff":
      handoffBox.textContent = JSON.stringify(msg.payload, null, 2);
      break;
    case "error":
      addMessage(`Error: ${msg.payload.message}`, "system");
      break;
    case "end":
      addMessage("Session ended.", "system");
      break;
  }
});

textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  ws.send(JSON.stringify({ type: "text", text }));
  textInput.value = "";
});

micBtn.addEventListener("click", async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: "audio/webm" });
        const buffer = await blob.arrayBuffer();
        const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
        ws.send(JSON.stringify({ type: "audio", audio_base64: base64 }));
      };
      mediaRecorder.start();
      isRecording = true;
      micBtn.textContent = "⏹️ Stop recording";
      recordingIndicator.classList.remove("hidden");
    } catch (err) {
      addMessage(`Microphone permission denied or unavailable: ${err}`, "system");
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    micBtn.textContent = "🎙️ Start recording";
    recordingIndicator.classList.add("hidden");
  }
});
