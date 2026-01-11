/**
 * Real Estate AI Chat - JavaScript Controller
 * Handles chat interactions with RAG-powered backend
 * 
 * RELIABILITY FIXES:
 * - Timeout protection (45s) to prevent indefinite hangs
 * - Null/empty response handling with user-friendly fallbacks
 * - Never shows raw API or timeout errors to users
 * - Loading state always cleared in finally block
 */

class ChatManager {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatForm = document.getElementById('chatForm');
        this.chatInput = document.getElementById('chatInput');
        this.sendChatBtn = document.getElementById('sendChatBtn');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        
        this.isProcessing = false;
        
        // FIX: Reduced timeout to 45s for better UX (SQL queries return instantly)
        this.REQUEST_TIMEOUT = 45000;
        
        this.init();
    }
    
    init() {
        // Bind event listeners
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        
        // NOTE: Do NOT auto-focus chat input on page load
        // This causes browser to scroll down to AI Assistant section
        // Focus will be set when user interacts with the chat
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;
        
        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        this.chatInput.value = '';
        
        // Disable send button
        this.isProcessing = true;
        this.updateSendButton(true);
        
        // FIX: Add timeout controller to prevent indefinite hang on "Processing..."
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.REQUEST_TIMEOUT);
        
        try {
            // Send to backend with abort signal
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message }),
                signal: controller.signal
            });
            
            // FIX: Clear timeout immediately on response
            clearTimeout(timeoutId);
            
            // FIX: Handle non-JSON responses gracefully
            let data;
            try {
                data = await response.json();
            } catch (parseError) {
                console.error('Failed to parse response:', parseError);
                this.addMessage(
                    'No relevant data found in the current dataset. Try asking about properties in Mumbai, Pune, or Delhi.',
                    'assistant',
                    'parse_error'
                );
                return;
            }
            
            if (data.success) {
                // FIX: Handle empty, null, or whitespace-only responses gracefully
                const responseText = (data.response && typeof data.response === 'string' && data.response.trim().length > 0)
                    ? data.response 
                    : 'No relevant data found in the current dataset. Try asking about properties in Mumbai, Pune, or Delhi.';
                this.addMessage(responseText, 'assistant', data.source);
            } else {
                // FIX: Handle backend errors with user-friendly messages (never show raw errors)
                this.handleBackendError(response.status, data.error);
            }
        } catch (error) {
            // FIX: Clear timeout on any error
            clearTimeout(timeoutId);
            console.error('Chat error:', error);
            
            // FIX: User-friendly error messages (never show technical details)
            this.handleNetworkError(error);
        } finally {
            // FIX: ALWAYS re-enable send button - this prevents permanent "Processing..." state
            this.isProcessing = false;
            this.updateSendButton(false);
            this.chatInput.focus();
        }
    }
    
    /**
     * FIX: Centralized backend error handling with user-friendly messages
     */
    handleBackendError(status, errorMsg) {
        let userMessage;
        let source;
        
        switch (status) {
            case 503:
                // RAG unavailable
                userMessage = 'The AI assistant is currently initializing. Please try again in a moment or run: python run_rag.py';
                source = 'setup_required';
                break;
            case 404:
                // No results
                userMessage = 'No relevant data found in the current dataset. Try asking about properties in Mumbai, Pune, Delhi, or other major cities.';
                source = 'no_results';
                break;
            case 429:
                // Rate limited
                userMessage = 'The service is experiencing high demand. Please try again in a few seconds.';
                source = 'rate_limited';
                break;
            default:
                // Generic fallback - never show raw error
                userMessage = 'No relevant data found in the current dataset. Try asking about properties in Mumbai, Pune, or Delhi.';
                source = 'fallback';
        }
        
        this.addMessage(userMessage, 'assistant', source);
    }
    
    /**
     * FIX: Centralized network error handling with user-friendly messages
     */
    handleNetworkError(error) {
        let userMessage;
        let source;
        
        if (error.name === 'AbortError') {
            // Timeout - suggest simpler query
            userMessage = 'The request is taking longer than expected. Try a simpler query like:\n• "mumbai"\n• "locations in pune"\n• "average price in delhi"';
            source = 'timeout';
        } else if (error.message && error.message.includes('Failed to fetch')) {
            // Server not running
            userMessage = 'Unable to connect to the server. Please ensure the application is running.';
            source = 'connection_error';
        } else {
            // Generic network error - user-friendly message
            userMessage = 'No relevant data found in the current dataset. Try asking about properties in Mumbai, Pune, Delhi, or other major cities.';
            source = 'network_error';
        }
        
        this.addMessage(userMessage, 'assistant', source);
    }
    
    addMessage(text, sender = 'assistant', source = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex items-start space-x-3 animate-fade-in';
        
        if (sender === 'user') {
            // User message
            messageDiv.classList.add('justify-end');
            messageDiv.innerHTML = `
                <div class="flex-1 flex justify-end">
                    <div class="bg-gradient-to-r from-violet-600 to-indigo-600 rounded-lg shadow-lg shadow-violet-500/25 p-4 max-w-md">
                        <p class="text-sm text-white">${this.escapeHtml(text)}</p>
                    </div>
                </div>
                <div class="flex-shrink-0 w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center text-slate-300 text-sm">
                    <i class="fas fa-user"></i>
                </div>
            `;
        } else {
            // Assistant message
            const sourceHtml = source && source !== 'error' && source !== 'basic' ? `
                <div class="mt-2 pt-2 border-t border-slate-600">
                    <p class="text-xs text-slate-400">
                        <i class="fas fa-database mr-1"></i>
                        <span class="font-semibold">Source:</span> ${this.escapeHtml(source)}
                    </p>
                </div>
            ` : '';
            
            messageDiv.innerHTML = `
                <div class="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-indigo-500 flex items-center justify-center text-white text-sm">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="flex-1">
                    <div class="bg-slate-700/80 rounded-lg shadow-sm p-4 border border-slate-600 max-w-2xl">
                        <p class="text-sm text-slate-200 whitespace-pre-line">${this.escapeHtml(text)}</p>
                        ${sourceHtml}
                    </div>
                </div>
            `;
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    clearChat() {
        // Remove all messages except the welcome message
        const messages = this.chatMessages.querySelectorAll('.animate-fade-in');
        messages.forEach((msg, index) => {
            if (index > 0) { // Keep first message (welcome)
                msg.remove();
            }
        });
        
        this.chatInput.focus();
    }
    
    updateSendButton(isLoading) {
        if (isLoading) {
            this.sendChatBtn.disabled = true;
            this.sendChatBtn.innerHTML = `
                <span>Processing...</span>
                <i class="fas fa-spinner fa-spin"></i>
            `;
            this.sendChatBtn.classList.add('opacity-70', 'cursor-not-allowed');
        } else {
            this.sendChatBtn.disabled = false;
            this.sendChatBtn.innerHTML = `
                <span>Send</span>
                <i class="fas fa-paper-plane"></i>
            `;
            this.sendChatBtn.classList.remove('opacity-70', 'cursor-not-allowed');
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const chatManager = new ChatManager();
    console.log('Chat manager initialized');
});
