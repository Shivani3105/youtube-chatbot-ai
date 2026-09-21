/**
 * TubeTalk AI - Frontend Client Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const videoUrlInput = document.getElementById('video-url-input');
    const clearUrlBtn = document.getElementById('clear-url-btn');
    const loadVideoBtn = document.getElementById('load-video-btn');
    const videoAlert = document.getElementById('video-alert');
    const playerPlaceholder = document.getElementById('player-placeholder');
    const youtubeIframe = document.getElementById('youtube-iframe');
    const videoStatsCard = document.getElementById('video-stats-card');
    
    // Stats Elements
    const statVideoId = document.getElementById('stat-video-id');
    const statChunks = document.getElementById('stat-chunks');
    const statLang = document.getElementById('stat-lang');
    const transcriptSnippetText = document.getElementById('transcript-snippet-text');
    
    // Chat Elements
    const chatStream = document.getElementById('chat-stream');
    const chatSubtitle = document.getElementById('chat-subtitle');
    const chatTextarea = document.getElementById('chat-textarea');
    const sendMsgBtn = document.getElementById('send-msg-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const quickChips = document.querySelectorAll('.chip');
    const serverStatus = document.getElementById('server-status');

    let currentVideoId = null;
    let isProcessing = false;

    // Check Backend Server Connection
    checkServerStatus();

    // Event Listeners
    clearUrlBtn.addEventListener('click', () => {
        videoUrlInput.value = '';
        videoUrlInput.focus();
    });

    loadVideoBtn.addEventListener('click', handleLoadVideo);

    videoUrlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleLoadVideo();
        }
    });

    chatTextarea.addEventListener('input', () => {
        // Auto-resize textarea
        chatTextarea.style.height = 'auto';
        chatTextarea.style.height = (chatTextarea.scrollHeight) + 'px';
        
        // Enable/disable send button
        sendMsgBtn.disabled = !chatTextarea.value.trim() || isProcessing;
    });

    chatTextarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendMsgBtn.disabled) {
                handleSendQuestion();
            }
        }
    });

    sendMsgBtn.addEventListener('click', handleSendQuestion);

    clearChatBtn.addEventListener('click', () => {
        if (confirm('Clear chat history?')) {
            resetChatStream();
        }
    });

    // Quick Chips Listener
    quickChips.forEach(chip => {
        chip.addEventListener('click', async () => {
            const promptText = chip.getAttribute('data-prompt');
            if (!currentVideoId) {
                const urlVal = videoUrlInput.value.trim();
                if (urlVal) {
                    await handleLoadVideo();
                } else {
                    showAlert('Please paste a YouTube video URL first.', 'error');
                    return;
                }
            }
            if (currentVideoId) {
                chatTextarea.value = promptText;
                chatTextarea.dispatchEvent(new Event('input'));
                handleSendQuestion();
            }
        });
    });

    // ==========================================
    // API & Event Handlers
    // ==========================================

    async function checkServerStatus() {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                serverStatus.innerHTML = '<span class="status-dot online"></span> API Server Ready';
                if (data.video_loaded) {
                    currentVideoId = data.video_id;
                    enableChatInput();
                }
            }
        } catch (err) {
            serverStatus.innerHTML = '<span class="status-dot" style="background:#EF4444"></span> Offline';
        }
    }

    function extractVideoId(urlOrId) {
        urlOrId = urlOrId.trim();
        if (urlOrId.length === 11 && !urlOrId.includes('/') && !urlOrId.includes('.')) {
            return urlOrId;
        }
        const match = urlOrId.match(/(?:v=|\/)([0-9A-Za-z_-]{11})/);
        return match ? match[1] : null;
    }

    async function handleLoadVideo() {
        const urlValue = videoUrlInput.value.trim();
        if (!urlValue) {
            showAlert('Please enter a YouTube video URL or ID.', 'error');
            return;
        }

        const extractedId = extractVideoId(urlValue);
        if (!extractedId) {
            showAlert('Invalid YouTube URL or Video ID format.', 'error');
            return;
        }

        // Set Loading State
        hideAlert();
        loadVideoBtn.disabled = true;
        loadVideoBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Fetching Transcript...';

        try {
            const response = await fetch('/api/load-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_url: urlValue })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to fetch transcript.');
            }

            // Success
            currentVideoId = data.video_id;
            
            // Update Video Preview
            playerPlaceholder.classList.add('hidden');
            youtubeIframe.src = `https://www.youtube.com/embed/${currentVideoId}`;
            youtubeIframe.classList.remove('hidden');

            // Update Metadata Stats
            statVideoId.textContent = currentVideoId;
            statChunks.textContent = `${data.chunk_count} chunks`;
            statLang.textContent = (data.language || 'en').toUpperCase();
            transcriptSnippetText.textContent = data.transcript_snippet;
            videoStatsCard.classList.remove('hidden');

            // Update Chat Header & Enable Input
            chatSubtitle.textContent = `Loaded Video ID: ${currentVideoId}`;
            enableChatInput();
            showAlert('Transcript indexed & FAISS vector store created!', 'success');

            // Remove Welcome Card if present
            const welcomeCard = chatStream.querySelector('.welcome-card');
            if (welcomeCard) {
                welcomeCard.remove();
            }

            // Add Assistant Greeting Message
            appendMessage('assistant', `Video transcript loaded successfully! Created **${data.chunk_count} text chunks** in vector store. Ask me any question about this video.`);

        } catch (error) {
            showAlert(error.message, 'error');
        } finally {
            loadVideoBtn.disabled = false;
            loadVideoBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> <span>Analyze & Fetch Transcript</span>';
        }
    }

    async function handleSendQuestion() {
        const questionText = chatTextarea.value.trim();
        if (!questionText || isProcessing) return;

        if (!currentVideoId) {
            showAlert('Please load a video first.', 'error');
            return;
        }

        // Add User Message
        appendMessage('user', questionText);
        
        // Reset Textarea
        chatTextarea.value = '';
        chatTextarea.style.height = 'auto';
        sendMsgBtn.disabled = true;
        isProcessing = true;

        // Add Loading Shimmer Indicator
        const loadingId = addLoadingIndicator();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: questionText })
            });

            const data = await response.json();

            // Remove loading indicator
            removeLoadingIndicator(loadingId);

            if (!response.ok) {
                throw new Error(data.detail || 'Error generating answer.');
            }

            // Append Assistant Answer
            appendMessage('assistant', data.answer);

        } catch (error) {
            removeLoadingIndicator(loadingId);
            appendMessage('assistant', `⚠️ **Error:** ${error.message}`);
        } finally {
            isProcessing = false;
            sendMsgBtn.disabled = !chatTextarea.value.trim();
        }
    }

    // ==========================================
    // UI Helper Functions
    // ==========================================

    function enableChatInput() {
        chatTextarea.disabled = false;
        chatTextarea.placeholder = "Ask a question about the video transcript...";
        chatTextarea.focus();
    }

    function showAlert(msg, type = 'success') {
        videoAlert.className = `alert-box ${type}`;
        videoAlert.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i> ${msg}`;
        videoAlert.classList.remove('hidden');
    }

    function hideAlert() {
        videoAlert.classList.add('hidden');
    }

    function formatMarkdown(text) {
        if (!text) return '';
        // Escape HTML
        let formatted = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
            
        // Bold formatting
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Bullet points
        formatted = formatted.replace(/^\*\s(.*)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/^-\s(.*)$/gm, '<li>$1</li>');
        
        // Line breaks
        formatted = formatted.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
        return `<p>${formatted}</p>`;
    }

    function appendMessage(role, text) {
        const msgRow = document.createElement('div');
        msgRow.className = `message-row ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = formatMarkdown(text);

        msgRow.appendChild(avatar);
        msgRow.appendChild(bubble);

        chatStream.appendChild(msgRow);
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const msgRow = document.createElement('div');
        msgRow.className = 'message-row assistant';
        msgRow.id = id;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'bubble loading-bubble';
        bubble.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;

        msgRow.appendChild(avatar);
        msgRow.appendChild(bubble);
        chatStream.appendChild(msgRow);
        scrollToBottom();
        return id;
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatStream.scrollTop = chatStream.scrollHeight;
    }

    function resetChatStream() {
        chatStream.innerHTML = `
            <div class="welcome-card">
                <div class="welcome-icon">
                    <i class="fa-solid fa-robot"></i>
                </div>
                <h3>Welcome to TubeTalk AI!</h3>
                <p>I can read YouTube transcripts, organize them into FAISS vector embeddings, and answer any questions grounded in the video's content.</p>
            </div>
        `;
    }
});
