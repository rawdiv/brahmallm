/**
 * Brahma LLM Platform - Main JavaScript
 * Handles authentication, API interactions, and UI updates
 */

document.addEventListener('DOMContentLoaded', function() {
    // Check authentication status and update UI
    checkAuthStatus();
    
    // Set up logout functionality
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function(e) {
            e.preventDefault();
            logout();
        });
    }
    
    // Setup page-specific initialization
    const currentPath = window.location.pathname;
    
    // Highlight current nav item
    highlightCurrentNavItem(currentPath);
});

/**
 * Check if user is authenticated and update UI accordingly
 */
function checkAuthStatus() {
    const token = localStorage.getItem('access_token');
    const user = JSON.parse(localStorage.getItem('user') || 'null');
    
    const authButtons = document.getElementById('auth-buttons');
    const userDropdown = document.getElementById('user-dropdown');
    const usernameSpan = document.getElementById('username');
    
    if (token && user) {
        // User is logged in
        if (authButtons) authButtons.classList.add('d-none');
        if (userDropdown) {
            userDropdown.classList.remove('d-none');
            if (usernameSpan) usernameSpan.textContent = user.username;
        }
        
        // If on login or register page, redirect to dashboard
        if (window.location.pathname === '/login' || window.location.pathname === '/register') {
            window.location.href = '/dashboard';
        }
    } else {
        // User is not logged in
        if (authButtons) authButtons.classList.remove('d-none');
        if (userDropdown) userDropdown.classList.add('d-none');
        
        // List of protected pages that require login
        const protectedPages = ['/dashboard', '/account', '/api-keys', '/playground', '/chat'];
        
        // If on a protected page, redirect to login
        if (protectedPages.some(page => window.location.pathname.startsWith(page))) {
            window.location.href = '/login';
        }
    }
}

/**
 * Logout the user
 */
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    
    // Show a toast or alert
    showAlert('You have been logged out successfully', 'success');
    
    // Redirect to home page after a short delay
    setTimeout(() => {
        window.location.href = '/';
    }, 1000);
}

/**
 * Show an alert message to the user
 * @param {string} message - The message to display
 * @param {string} type - The alert type (success, danger, warning, info)
 */
function showAlert(message, type) {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => {
            if (alertContainer.contains(alert)) {
                alertContainer.removeChild(alert);
            }
        }, 150);
    }, 5000);
}

/**
 * Highlight the current nav item based on the current page
 * @param {string} currentPath - The current page path
 */
function highlightCurrentNavItem(currentPath) {
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Map of paths to nav item IDs
    const pathToNavMap = {
        '/': 'nav-home',
        '/chat': 'nav-chat',
        '/playground': 'nav-playground',
        '/dashboard': 'nav-dashboard',
        '/docs': 'nav-docs'
    };
    
    // Check exact matches first
    if (pathToNavMap[currentPath]) {
        const navItem = document.getElementById(pathToNavMap[currentPath]);
        if (navItem) {
            navItem.querySelector('.nav-link').classList.add('active');
            return;
        }
    }
    
    // Check for partial matches
    for (const [path, navId] of Object.entries(pathToNavMap)) {
        if (path !== '/' && currentPath.startsWith(path)) {
            const navItem = document.getElementById(navId);
            if (navItem) {
                navItem.querySelector('.nav-link').classList.add('active');
                return;
            }
        }
    }
}

/**
 * Make an authenticated API request
 * @param {string} url - The API endpoint URL
 * @param {Object} options - Request options
 * @returns {Promise} - The fetch promise
 */
async function apiRequest(url, options = {}) {
    const token = localStorage.getItem('access_token');
    
    if (!token) {
        throw new Error('Not authenticated');
    }
    
    const headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`
    };
    
    const response = await fetch(url, {
        ...options,
        headers
    });
    
    if (response.status === 401) {
        // Token expired or invalid
        localStorage.removeItem('access_token');
        localStorage.removeItem('user');
        window.location.href = '/login?session_expired=true';
        throw new Error('Session expired');
    }
    
    return response;
}

/**
 * Format a date string
 * @param {string} dateString - ISO date string
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    if (!dateString) return 'Never';
    
    const date = new Date(dateString);
    return date.toLocaleString();
}

/**
 * Truncate a string to a specific length and add ellipsis if needed
 * @param {string} str - The string to truncate
 * @param {number} maxLength - The maximum length
 * @returns {string} - The truncated string
 */
function truncateString(str, maxLength) {
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength) + '...';
}

/**
 * Copy text to clipboard
 * @param {string} text - The text to copy
 * @returns {Promise} - Promise that resolves when text is copied
 */
function copyToClipboard(text) {
    return navigator.clipboard.writeText(text);
}

/**
 * Handle form submission with JSON data
 * @param {HTMLFormElement} form - The form element
 * @param {string} url - The API endpoint URL
 * @param {string} method - The HTTP method
 * @returns {Promise} - Promise that resolves with the response data
 */
async function handleFormSubmit(form, url, method = 'POST') {
    const formData = new FormData(form);
    const jsonData = {};
    
    for (const [key, value] of formData.entries()) {
        jsonData[key] = value;
    }
    
    const response = await apiRequest(url, {
        method,
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(jsonData)
    });
    
    return response.json();
}
