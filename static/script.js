document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    
    // Floating buttons & modal elements
    const floatingActions = document.getElementById('floating-actions');
    const btnReset = document.getElementById('btn-reset');
    const btnExit = document.getElementById('btn-exit');
    const exitModal = document.getElementById('exit-modal');
    const btnConfirmExit = document.getElementById('btn-confirm-exit');
    const btnCancelExit = document.getElementById('btn-cancel-exit');

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value === '') {
            this.style.height = 'auto';
        }
    });

    // Handle enter to submit
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;

        // Reset input
        userInput.value = '';
        userInput.style.height = 'auto';

        // Switch layout to chat mode if it's the first message
        if (appContainer.classList.contains('landing-mode')) {
            appContainer.classList.remove('landing-mode');
            appContainer.classList.add('chat-mode');
            floatingActions.classList.remove('hidden'); // Show floating buttons
        }

        // Add User Message to UI
        appendMessage(message, 'user');

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Remove typing indicator
            removeElement(typingId);

            if (response.ok) {
                // Format response (replace newlines with <br> for HTML)
                const formattedResponse = data.response.replace(/\n/g, '<br>');
                appendMessage(formattedResponse, 'bot', true);

                // Show professional counselor button if flagged
                if (data.show_professional_button) {
                    appendProfessionalButton();
                }

                // Handle session ended — lock input and show end banner
                if (data.session_ended) {
                    lockSessionInput();
                    appendSessionEndBanner();
                }
            } else {
                appendMessage("Error: " + (data.error || "Gagal menghubungi server."), 'bot');
            }
            
        } catch (error) {
            removeElement(typingId);
            appendMessage("Terjadi kesalahan jaringan. Silakan periksa koneksi Anda.", 'bot');
            console.error('Error:', error);
        }
    });

    // Utility: Append message to chat history
    function appendMessage(text, sender, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-msg`;
        
        if (isHtml) {
            msgDiv.innerHTML = text;
        } else {
            msgDiv.textContent = text;
        }
        
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    // Utility: Append professional counselor button to chat
    function appendProfessionalButton() {
        const btnContainer = document.createElement('div');
        btnContainer.className = 'professional-button-container';

        const btn = document.createElement('button');
        btn.id = 'btn-professional-counselor';
        btn.className = 'professional-button';
        btn.innerHTML = '💬 Hubungi Konselor Profesional';
        btn.addEventListener('click', () => {
            // Dummy action — will be replaced with live-chat integration later
            alert('Fitur live-chat dengan konselor profesional akan segera tersedia. Silakan hubungi layanan konseling kampus Anda untuk saat ini.');
        });

        btnContainer.appendChild(btn);
        chatHistory.appendChild(btnContainer);
        scrollToBottom();
    }

    // Utility: Show typing dots
    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-msg typing-indicator';
        typingDiv.id = id;
        
        typingDiv.innerHTML = `
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        `;
        
        chatHistory.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    // Utility: Remove element by ID
    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // Utility: Lock input after session ends
    function lockSessionInput() {
        userInput.disabled = true;
        userInput.placeholder = 'Sesi konseling telah selesai.';
        document.getElementById('send-btn').disabled = true;
    }

    // Utility: Unlock input (called on reset)
    function unlockSessionInput() {
        userInput.disabled = false;
        userInput.placeholder = 'Ceritakan perasaanmu di sini...';
        document.getElementById('send-btn').disabled = false;
    }

    // Utility: Append session end banner to chat
    function appendSessionEndBanner() {
        const banner = document.createElement('div');
        banner.className = 'session-end-banner';
        banner.innerHTML = `
            <p>✅ <strong>Sesi konseling selesai.</strong></p>
            <p>Untuk memulai sesi baru, klik tombol <strong>Reset</strong> (↺) di pojok kanan atas.<br>
               Untuk menutup aplikasi, klik tombol <strong>Keluar</strong> (🚪).</p>
        `;
        chatHistory.appendChild(banner);
        scrollToBottom();
    }

    // Utility: Scroll to bottom of chat
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Call /reset on load just to ensure a fresh session 
    // (useful if user refreshes the page)
    fetch('/reset', { method: 'POST' }).catch(e => console.log('Reset error', e));

    // --- Floating Buttons Logic ---
    
    // Reset Chat
    btnReset.addEventListener('click', async () => {
        try {
            await fetch('/reset', { method: 'POST' });
            chatHistory.innerHTML = ''; // Clear chat
            appContainer.classList.remove('chat-mode');
            appContainer.classList.add('landing-mode');
            floatingActions.classList.add('hidden'); // Hide floating buttons
            unlockSessionInput(); // Re-enable input after session reset
        } catch (e) {
            console.error('Failed to reset:', e);
        }
    });

    // Show Exit Modal
    btnExit.addEventListener('click', () => {
        exitModal.classList.remove('hidden');
    });

    // Hide Exit Modal (Cancel)
    btnCancelExit.addEventListener('click', () => {
        exitModal.classList.add('hidden');
    });

    // Confirm Exit (Shutdown)
    btnConfirmExit.addEventListener('click', async () => {
        exitModal.classList.add('hidden');
        try {
            await fetch('/shutdown', { method: 'POST' });
            // Attempt to close window (works if opened via script, otherwise browsers block it)
            window.close();
            
            // Fallback if window.close() is blocked
            document.body.innerHTML = `
                <div style="display:flex; justify-content:center; align-items:center; height:100vh; flex-direction:column; font-family:sans-serif; text-align:center; padding:20px;">
                    <h1 style="color:#E63946;">Sesi Berakhir</h1>
                    <p style="color:#333;">Program Python telah dimatikan. Anda dapat menutup tab ini secara manual.</p>
                </div>
            `;
        } catch (e) {
            console.error('Failed to shutdown:', e);
            alert('Gagal menghentikan server.');
        }
    });
});
