// ── State ────────────────────────────────────────────────────
let currentSessionId = null;

// ── DOM Elements ────────────────────────────────────────────
const sessionsList = document.getElementById("sessions-list");
const chatMessages = document.getElementById("chat-messages");
const chatInputBar = document.getElementById("chat-input-bar");
const welcomeScreen = document.getElementById("welcome-screen");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const uploadBtn = document.getElementById("upload-btn");
const uploadModal = document.getElementById("upload-modal");
const closeModal = document.getElementById("close-modal");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const docCount = document.getElementById("doc-count");


// ── API Helpers ─────────────────────────────────────────────

async function api(method, path, body = null) {
    const opts = { method, headers: {} };
    if (body && !(body instanceof FormData)) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
    } else if (body) {
        opts.body = body;
    }
    const res = await fetch(`/api${path}`, opts);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(err.detail || "Request failed");
    }
    return res.json();
}


// ── Sessions ────────────────────────────────────────────────

async function loadSessions() {
    const sessions = await api("GET", "/sessions");
    sessionsList.innerHTML = "";
    sessions.forEach((s) => {
        const div = document.createElement("div");
        div.className = "session-item" + (s.id === currentSessionId ? " active" : "");
        div.innerHTML = `
            <span class="session-title">${escapeHtml(s.title)}</span>
            <button class="delete-session" title="Delete">&times;</button>
        `;
        div.querySelector(".session-title").addEventListener("click", () => openSession(s.id));
        div.querySelector(".delete-session").addEventListener("click", async (e) => {
            e.stopPropagation();
            await api("DELETE", `/sessions/${s.id}`);
            if (currentSessionId === s.id) {
                currentSessionId = null;
                showWelcome();
            }
            loadSessions();
        });
        sessionsList.appendChild(div);
    });
}

async function openSession(sessionId) {
    currentSessionId = sessionId;
    const data = await api("GET", `/sessions/${sessionId}`);
    showChat();
    chatMessages.innerHTML = "";
    data.messages.forEach((m) => appendMessage(m.role, m.content, m.sources));
    scrollToBottom();
    loadSessions();
}

newChatBtn.addEventListener("click", async () => {
    const session = await api("POST", "/sessions", { title: "New Chat" });
    currentSessionId = session.id;
    showChat();
    chatMessages.innerHTML = "";
    loadSessions();
});


// ── Chat ────────────────────────────────────────────────────

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !currentSessionId) return;

    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    appendMessage("user", message);
    const typingEl = showTyping();
    scrollToBottom();

    try {
        const data = await api("POST", "/chat", {
            session_id: currentSessionId,
            message: message,
        });
        removeTyping(typingEl);
        const sourcesJson = data.sources ? JSON.stringify(data.sources) : null;
        appendMessage("assistant", data.answer, sourcesJson);
        loadSessions();  // refresh titles
    } catch (err) {
        removeTyping(typingEl);
        appendMessage("assistant", `Error: ${err.message}`);
    }

    sendBtn.disabled = false;
    scrollToBottom();
}

sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});


// ── Message Rendering ───────────────────────────────────────

function appendMessage(role, content, sourcesStr = null) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    let html = `<div class="message-bubble">${escapeHtml(content)}</div>`;

    if (sourcesStr && role === "assistant") {
        try {
            const sources = typeof sourcesStr === "string" ? JSON.parse(sourcesStr) : sourcesStr;
            if (sources.length > 0) {
                const uniqueSources = [...new Set(sources.map((s) => s.source))];
                html += `<div class="message-sources">📎 Sources: ${uniqueSources.join(", ")}</div>`;
            }
        } catch (e) { /* ignore parse errors */ }
    }

    div.innerHTML = html;
    chatMessages.appendChild(div);
}

function showTyping() {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ── View Switching ──────────────────────────────────────────

function showChat() {
    welcomeScreen.style.display = "none";
    chatMessages.style.display = "block";
    chatInputBar.style.display = "flex";
    chatInput.focus();
}

function showWelcome() {
    welcomeScreen.style.display = "flex";
    chatMessages.style.display = "none";
    chatInputBar.style.display = "none";
}


// ── Upload Modal ────────────────────────────────────────────

uploadBtn.addEventListener("click", () => {
    uploadModal.style.display = "flex";
    loadDocuments();
});

closeModal.addEventListener("click", () => {
    uploadModal.style.display = "none";
});

uploadModal.addEventListener("click", (e) => {
    if (e.target === uploadModal) uploadModal.style.display = "none";
});

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
});

async function handleFiles(files) {
    for (const file of files) {
        uploadStatus.innerHTML = `<span class="upload-loading">Uploading & processing ${escapeHtml(file.name)}...</span>`;
        try {
            const form = new FormData();
            form.append("file", file);
            const result = await api("POST", "/upload", form);
            uploadStatus.innerHTML = `<span class="upload-success">✓ ${escapeHtml(result.message)}</span>`;
        } catch (err) {
            uploadStatus.innerHTML = `<span class="upload-error">✗ ${escapeHtml(err.message)}</span>`;
        }
    }
    loadDocuments();
    loadStatus();
}

async function loadDocuments() {
    try {
        const docs = await api("GET", "/documents");
        docList.innerHTML = "";
        if (docs.length === 0) {
            docList.innerHTML = '<p style="color:#999;font-size:13px;">No documents uploaded yet.</p>';
            return;
        }
        docs.forEach((doc) => {
            const div = document.createElement("div");
            div.className = "doc-item";
            div.innerHTML = `
                <div class="doc-info">
                    <span>${escapeHtml(doc.filename)}</span>
                    <span class="doc-chunks">${doc.chunk_count} chunks</span>
                </div>
                <button class="delete-doc" title="Delete">🗑</button>
            `;
            div.querySelector(".delete-doc").addEventListener("click", async () => {
                await api("DELETE", `/documents/${encodeURIComponent(doc.filename)}`);
                loadDocuments();
                loadStatus();
            });
            docList.appendChild(div);
        });
    } catch (err) {
        docList.innerHTML = `<p style="color:#dc2626;font-size:13px;">Failed to load documents.</p>`;
    }
}


// ── Status ──────────────────────────────────────────────────

async function loadStatus() {
    try {
        const status = await api("GET", "/status");
        docCount.textContent = `${status.total_documents} docs · ${status.total_chunks} chunks`;
    } catch (e) {
        docCount.textContent = "";
    }
}


// ── Utilities ───────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


// ── Init ────────────────────────────────────────────────────

loadSessions();
loadStatus();
