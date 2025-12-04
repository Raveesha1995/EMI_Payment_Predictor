// API Base URL
const API_BASE_URL = 'http://localhost:5000/api';

// Show loading indicator
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

// Hide loading indicator
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// Show error message
function showError(container, message) {
    container.innerHTML = `<div class="error">${message}</div>`;
    container.classList.add('show');
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// Predict single customer
async function predictSingle() {
    const customerId = document.getElementById('customer-id').value.trim();
    const useLLM = document.getElementById('use-llm').checked; // LLM is core feature, enabled by default
    const resultContainer = document.getElementById('single-result');
    
    if (!customerId) {
        showError(resultContainer, 'Please enter a Customer ID');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                customer_id: customerId,
                use_llm: useLLM
            })
        });
        
        const data = await response.json();
        hideLoading();
        
        if (!response.ok) {
            showError(resultContainer, data.error || 'Prediction failed');
            return;
        }
        
        displaySingleResult(data.prediction);
        
    } catch (error) {
        hideLoading();
        showError(resultContainer, `Error: ${error.message}. Make sure the backend server is running.`);
    }
}

// Display single prediction result
function displaySingleResult(prediction) {
    console.log('Prediction data received:', prediction); // Debug log
    const container = document.getElementById('single-result');
    const confidence = typeof prediction.confidence_score === 'number' 
        ? prediction.confidence_score 
        : parseFloat(prediction.confidence_score) || 0;
    const avgDelay = typeof prediction.average_delay === 'number' 
        ? prediction.average_delay 
        : parseFloat(prediction.average_delay) || 0;
    
    let confidenceClass = 'confidence-medium';
    if (confidence > 0.8) confidenceClass = 'confidence-high';
    if (confidence < 0.6) confidenceClass = 'confidence-low';
    
    let html = `
        <div class="prediction-result">
            <h3>Prediction Result</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Customer ID</th>
                            <th>Last Demand Date</th>
                            <th>Last Payment</th>
                            <th>Next Demand Date</th>
                            <th>Predicted Date</th>
                            <th>Avg Delay</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>${prediction.customer_id || 'N/A'}</td>
                            <td>${prediction.last_demand_date || 'N/A'}</td>
                            <td>${prediction.last_payment_date || 'N/A'}</td>
                            <td>${prediction.next_demand_date || 'N/A'}</td>
                            <td><strong style="color: #667eea;">${prediction.predicted_payment_date || 'N/A'}</strong></td>
                            <td>${avgDelay.toFixed(2)}</td>
                            <td class="${confidenceClass}">${(confidence * 100).toFixed(1)}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    if (prediction.llm_explanation) {
        html += `<div class="llm-explanation"><strong>LLM Explanation:</strong><br>${prediction.llm_explanation}</div>`;
    }
    
    container.innerHTML = html;
    container.classList.add('show');
}

// Predict batch
async function predictBatch() {
    const customerIdsInput = document.getElementById('customer-ids').value.trim();
    const useLLM = document.getElementById('use-llm-batch').checked; // LLM is core feature, enabled by default
    const resultContainer = document.getElementById('batch-result');
    
    let customerIds = [];
    if (customerIdsInput) {
        customerIds = customerIdsInput.split(',').map(id => id.trim()).filter(id => id);
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                customer_ids: customerIds.length > 0 ? customerIds : undefined,
                use_llm: useLLM
            })
        });
        
        const data = await response.json();
        hideLoading();
        
        if (!response.ok) {
            showError(resultContainer, data.error || 'Batch prediction failed');
            return;
        }
        
        displayBatchResult(data);
        
    } catch (error) {
        hideLoading();
        showError(resultContainer, `Error: ${error.message}. Make sure the backend server is running.`);
    }
}

// Display batch prediction results
function displayBatchResult(data) {
    const container = document.getElementById('batch-result');
    
    let html = `
        <div class="success">
            <strong>Batch Prediction Complete!</strong><br>
            Total predictions: ${data.count}
        </div>
    `;
    
    if (data.insights) {
        html += `<div class="llm-explanation"><strong>LLM Insights:</strong><br>${data.insights}</div>`;
    }
    
    html += `
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Customer ID</th>
                        <th>Last Demand Date</th>
                        <th>Last Payment</th>
                        <th>Next Demand Date</th>
                        <th>Predicted Date</th>
                        <th>Avg Delay</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    data.predictions.forEach(pred => {
        const confidence = typeof pred.confidence_score === 'number' 
            ? pred.confidence_score 
            : parseFloat(pred.confidence_score) || 0;
        const avgDelay = typeof pred.average_delay === 'number' 
            ? pred.average_delay 
            : parseFloat(pred.average_delay) || 0;
        
        let confidenceClass = 'confidence-medium';
        if (confidence > 0.8) confidenceClass = 'confidence-high';
        if (confidence < 0.6) confidenceClass = 'confidence-low';
        
        html += `
            <tr>
                <td>${pred.customer_id || 'N/A'}</td>
                <td>${pred.last_demand_date || 'N/A'}</td>
                <td>${pred.last_payment_date || 'N/A'}</td>
                <td>${pred.next_demand_date || 'N/A'}</td>
                <td><strong>${pred.predicted_payment_date || 'N/A'}</strong></td>
                <td>${avgDelay.toFixed(2)}</td>
                <td class="${confidenceClass}">${(confidence * 100).toFixed(1)}%</td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
    container.classList.add('show');
}

// Load customers list
async function loadCustomers() {
    const container = document.getElementById('customers-list');
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/customers`);
        const data = await response.json();
        hideLoading();
        
        if (!response.ok) {
            showError(container, data.error || 'Failed to load customers');
            return;
        }
        
        let html = `
            <div class="success">
                <strong>Total Customers: ${data.total}</strong>
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Customer ID</th>
                            <th>Total Payments</th>
                            <th>First Payment</th>
                            <th>Last Payment</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        data.customers.forEach(customer => {
            html += `
                <tr>
                    <td>${customer.customer_id}</td>
                    <td>${customer.total_payments}</td>
                    <td>${customer.first_payment}</td>
                    <td>${customer.last_payment}</td>
                    <td>
                        <button class="btn btn-primary" style="padding: 6px 12px; font-size: 0.9em;" 
                                onclick="predictCustomer('${customer.customer_id}')">
                            Predict
                        </button>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
        
        container.innerHTML = html;
        container.classList.add('show');
        
    } catch (error) {
        hideLoading();
        showError(container, `Error: ${error.message}. Make sure the backend server is running.`);
    }
}

// Predict for a customer from the list
function predictCustomer(customerId) {
    document.getElementById('customer-id').value = customerId;
    showTab('single');
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.textContent === 'Single Prediction') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    setTimeout(() => predictSingle(), 100);
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        if (data.model_loaded) {
            console.log('✓ Backend connected and model loaded');
        } else {
            console.warn('⚠ Backend connected but model not loaded');
        }
    } catch (error) {
        console.error('✗ Backend not available:', error.message);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    checkHealth();
});

