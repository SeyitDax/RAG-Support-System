/**
 * Frontend JavaScript for RAG Customer Support Demo
 * Handles chat interface, API communication, and real-time updates
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';
const POLLING_INTERVAL = 30000; // 30 seconds

// State management
let currentSessionId = generateSessionId();
let messageCounter = 0;
let systemStatus = {
    api: false,
    rag_engine: false,
    vector_store: false
};

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const sendButtonText = document.getElementById('sendButtonText');
const sendButtonSpinner = document.getElementById('sendButtonSpinner');
const statusIndicator = document.getElementById('statusIndicator');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const responseTime = document.getElementById('responseTime');
const confidenceScore = document.getElementById('confidenceScore');
const confidenceBar = document.getElementById('confidenceFill');
const confidenceExplanation = document.getElementById('confidenceExplanation');
const sourcesPanel = document.getElementById('sourcesPanel');
const sourcesList = document.getElementById('sourcesList');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkSystemHealth();
    
    // Set up periodic health checks
    setInterval(checkSystemHealth, POLLING_INTERVAL);
});

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    // Send button click
    sendButton.addEventListener('click', sendQuestion);
    
    // Enter key handling
    questionInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendQuestion();
        }
        if (event.key === 'Enter' && event.ctrlKey) {
            event.preventDefault();
            sendQuestion();
        }
    });
    
    // Auto-resize textarea
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

/**
 * Generate unique session ID
 */
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

/**
 * Check system health and update status indicators
 */
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === 'healthy') {
            updateSystemStatus(true, 'System Online - All components healthy', data.components);
        } else {
            updateSystemStatus(false, `System ${data.status || 'Unknown'} - Some issues detected`, data.components);
        }
        
        // Update system info panel
        updateSystemInfo(data);
        
    } catch (error) {
        updateSystemStatus(false, 'System Offline - Unable to connect to API');
        console.error('Health check failed:', error);
    }
}

/**
 * Update system status indicator
 */
function updateSystemStatus(isOnline, message, components = {}) {
    statusText.textContent = message;
    statusDot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
    
    // Update system status for UI decisions
    systemStatus = {
        api: isOnline,
        rag_engine: components?.rag_engine === 'healthy',
        vector_store: components?.vector_store === 'healthy'
    };
    
    // Update system info
    document.getElementById('apiStatus').textContent = isOnline ? 'Online' : 'Offline';
    document.getElementById('vectorStatus').textContent = components?.vector_store || '--';
}

/**
 * Update system information panel
 */
function updateSystemInfo(healthData) {
    if (healthData.components) {
        document.getElementById('apiStatus').textContent = 
            healthData.components.api === 'healthy' ? 'Online' : 'Error';
        document.getElementById('vectorStatus').textContent = 
            healthData.components.vector_store || 'Unknown';
    }
    
    // Try to get system stats for document count
    getSystemStats();
}

/**
 * Get detailed system statistics
 */
async function getSystemStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/system/stats`);
        if (response.ok) {
            const data = await response.json();
            if (data.vector_store && data.vector_store.total_vectors !== undefined) {
                document.getElementById('documentCount').textContent = 
                    data.vector_store.total_vectors.toLocaleString();
            }
        }
    } catch (error) {
        console.log('Stats unavailable:', error.message);
    }
}

/**
 * Send a question to the API
 */
async function sendQuestion() {
    const question = questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question');
        return;
    }
    
    if (!systemStatus.api) {
        showError('System is offline. Please wait for connection to be restored.');
        return;
    }
    
    // Disable input and show loading state
    setLoadingState(true);
    
    // Add user message to chat
    addMessage(question, 'user');
    
    // Clear input
    questionInput.value = '';
    questionInput.style.height = 'auto';
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: 3,
                session_id: currentSessionId,
                user_id: 'demo_user'
            })
        });
        
        const data = await response.json();
        const processingTime = Date.now() - startTime;
        
        if (response.ok && data.success) {
            // Add bot response
            addMessage(data.response, 'bot', {
                confidence: data.confidence,
                processingTime: data.processing_time || (processingTime / 1000),
                sources: data.sources || [],
                shouldEscalate: data.should_escalate,
                autoResponse: data.auto_response
            });
            
            // Update confidence display
            updateConfidenceDisplay(data.confidence);
            
            // Update sources
            updateSourcesDisplay(data.sources || []);
            
            // Update stats
            updateStats(data.processing_time || (processingTime / 1000), data.confidence);
            
        } else {
            // Handle API errors
            const errorMessage = data.response || data.error || 'Sorry, I encountered an error processing your question.';
            addMessage(errorMessage, 'bot', {
                confidence: 0,
                processingTime: processingTime / 1000,
                isError: true
            });
            
            updateStats(processingTime / 1000, 0);
        }
        
    } catch (error) {
        console.error('API request failed:', error);
        addMessage('Sorry, I\'m having trouble connecting to the server. Please try again later.', 'bot', {
            confidence: 0,
            processingTime: (Date.now() - startTime) / 1000,
            isError: true
        });
    } finally {
        setLoadingState(false);
    }
}

/**
 * Add a message to the chat
 */
function addMessage(content, sender, metadata = {}) {
    messageCounter++;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.id = `message-${messageCounter}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Format message content
    if (typeof content === 'string') {
        messageContent.innerHTML = formatMessageContent(content);
    } else {
        messageContent.appendChild(content);
    }
    
    // Add metadata for bot messages
    if (sender === 'bot' && (metadata.confidence !== undefined || metadata.processingTime)) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        const timeSpan = document.createElement('span');
        timeSpan.textContent = `${formatProcessingTime(metadata.processingTime || 0)}`;
        
        const confidenceSpan = document.createElement('span');
        if (metadata.confidence !== undefined && !metadata.isError) {
            confidenceSpan.className = `confidence-badge ${getConfidenceClass(metadata.confidence)}`;
            confidenceSpan.textContent = `${(metadata.confidence * 100).toFixed(1)}%`;
            
            // Add escalation indicator
            if (metadata.shouldEscalate) {
                const escalationSpan = document.createElement('span');
                escalationSpan.className = 'escalation-indicator';
                escalationSpan.textContent = '👤 Escalated';
                escalationSpan.style.marginLeft = '8px';
                escalationSpan.style.fontSize = '0.7rem';
                escalationSpan.style.color = 'var(--warning)';
                confidenceSpan.appendChild(escalationSpan);
            }
        }
        
        metaDiv.appendChild(timeSpan);
        if (confidenceSpan.textContent) {
            metaDiv.appendChild(confidenceSpan);
        }
        
        messageContent.appendChild(metaDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    // Add to chat (remove welcome message if this is the first real message)
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage && messageCounter === 1) {
        welcomeMessage.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Animate new message
    requestAnimationFrame(() => {
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        requestAnimationFrame(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });
    });
}

/**
 * Format message content (basic markdown-like formatting)
 */
function formatMessageContent(content) {
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
}

/**
 * Format processing time for display
 */
function formatProcessingTime(seconds) {
    if (seconds < 1) {
        return `${Math.round(seconds * 1000)}ms`;
    }
    return `${seconds.toFixed(2)}s`;
}

/**
 * Get CSS class for confidence level
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

/**
 * Update confidence display
 */
function updateConfidenceDisplay(confidence) {
    const percentage = Math.round(confidence * 100);
    
    // Update confidence bar
    confidenceBar.style.width = `${percentage}%`;
    
    // Update confidence score
    confidenceScore.textContent = `${percentage}%`;
    confidenceScore.className = getConfidenceClass(confidence);
}

/**
 * Update sources display
 */
function updateSourcesDisplay(sources) {
    if (!sources || sources.length === 0) {
        sourcesPanel.style.display = 'none';
        return;
    }
    
    sourcesPanel.style.display = 'block';
    sourcesList.innerHTML = '';
    
    sources.forEach(source => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source-item';
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'source-name';
        nameSpan.textContent = source.source;
        
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'source-score';
        scoreSpan.textContent = `${(source.relevance_score * 100).toFixed(1)}%`;
        
        sourceDiv.appendChild(nameSpan);
        sourceDiv.appendChild(scoreSpan);
        sourcesList.appendChild(sourceDiv);
    });
}

/**
 * Update stats display
 */
function updateStats(processingTime, confidence) {
    responseTime.textContent = formatProcessingTime(processingTime);
    
    if (confidence !== undefined) {
        confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;
        confidenceScore.className = getConfidenceClass(confidence);
    }
}

/**
 * Set loading state
 */
function setLoadingState(isLoading) {
    sendButton.disabled = isLoading;
    questionInput.disabled = isLoading;
    
    if (isLoading) {
        sendButtonText.style.display = 'none';
        sendButtonSpinner.style.display = 'block';
    } else {
        sendButtonText.style.display = 'block';
        sendButtonSpinner.style.display = 'none';
    }
}

/**
 * Set a pre-defined query in the input
 */
function setQuery(query) {
    questionInput.value = query;
    questionInput.focus();
    
    // Auto-resize textarea
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    
    // Scroll input into view on mobile
    if (window.innerWidth <= 768) {
        questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Create a temporary error message
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--error);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        z-index: 1000;
        font-weight: 500;
        max-width: 300px;
    `;
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Animate in
    errorDiv.style.opacity = '0';
    errorDiv.style.transform = 'translateY(-20px)';
    requestAnimationFrame(() => {
        errorDiv.style.transition = 'all 0.3s ease';
        errorDiv.style.opacity = '1';
        errorDiv.style.transform = 'translateY(0)';
    });
    
    // Remove after 3 seconds
    setTimeout(() => {
        errorDiv.style.opacity = '0';
        errorDiv.style.transform = 'translateY(-20px)';
        setTimeout(() => errorDiv.remove(), 300);
    }, 3000);
}

/**
 * Utility function for debouncing
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global access (example queries)
window.setQuery = setQuery;
window.sendQuestion = sendQuestion;