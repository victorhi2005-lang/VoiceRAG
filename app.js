let mediaRecorder;
let audioChunks = [];

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const uploadBtn = document.getElementById('uploadBtn');
const audioPlayback = document.getElementById('audioPlayback');

startBtn.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

    mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        audioPlayback.src = URL.createObjectURL(blob);

        uploadBtn.audioBlob = blob;
        uploadBtn.disabled = false;
    };

    mediaRecorder.start();
    startBtn.disabled = true;
    stopBtn.disabled = false;
};

stopBtn.onclick = () => {
    mediaRecorder.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
};

uploadBtn.onclick = async () => {
    const formData = new FormData();
    formData.append("file", uploadBtn.audioBlob, "recording.webm");

    const res = await fetch("http://127.0.0.1:8000/upload-audio/", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    alert("上傳成功！");
    console.log(data);
};

document.getElementById("askBtn").onclick = async () => {
    const question = document.getElementById("question").value;

    const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question })
    });

    const data = await res.json();
    document.getElementById("answer").innerText = data.answer;
};