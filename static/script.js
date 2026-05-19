document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');

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

    // Utility: Scroll to bottom of chat
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Call /reset on load just to ensure a fresh session 
    // (useful if user refreshes the page)
    fetch('/reset', { method: 'POST' }).catch(e => console.log('Reset error', e));
});
