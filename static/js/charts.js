/**
 * Chart.js Configuration for Real Estate Dashboard
 * Handles all chart visualizations including ROI, Buy vs Rent, and Price Distribution
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if chartData is available (passed from Flask template)
    if (typeof chartData === 'undefined') {
        console.warn('Chart data not available - charts.js loaded on non-dashboard page');
        return;
    }

    console.log('📊 Chart data received:', {
        roi_labels: chartData.roi_labels?.length || 0,
        price_distribution: chartData.price_distribution_labels?.length || 0,
        price_per_sqft: chartData.price_per_sqft_labels?.length || 0,
        city_properties: chartData.city_properties_labels?.length || 0,
        location_price: chartData.location_price_labels?.length || 0
    });

    // Initialize all charts with error handling
    // Each chart is wrapped in try/catch to prevent one failure from blocking others
    try { initROIChart(); console.log('✅ ROI Chart initialized'); } 
    catch(e) { console.error('❌ ROI Chart error:', e); }
    
    try { initBuyVsRentChart(); console.log('✅ Buy vs Rent Chart initialized'); } 
    catch(e) { console.error('❌ Buy vs Rent Chart error:', e); }
    
    try { initPriceDistributionChart(); console.log('✅ Price Distribution Chart initialized'); } 
    catch(e) { console.error('❌ Price Distribution Chart error:', e); }
    
    try { initCityPropertiesChart(); console.log('✅ City Properties Chart initialized'); } 
    catch(e) { console.error('❌ City Properties Chart error:', e); }
    
    try { initLocationPriceChart(); console.log('✅ Location Price Chart initialized'); } 
    catch(e) { console.error('❌ Location Price Chart error:', e); }
    
    try { initPricePerSqftChart(); console.log('✅ Price per Sqft Chart initialized'); } 
    catch(e) { console.error('❌ Price per Sqft Chart error:', e); }
});

/**
 * Show "No Data Available" message in a chart container
 * Displays contextual messages based on chart type
 */
function showNoDataMessage(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    // Contextual messages based on chart type
    const messages = {
        'roiChart': 'Buy vs Rent comparison data loading. Apply filters to see insights.',
        'buyVsRentChart': 'Buy vs Rent analysis not available for current selection.',
        'priceDistributionChart': 'No price data available. Try selecting different filters.',
        'cityPropertiesChart': 'No city data available in current view.',
        'locationPriceChart': 'Location prices not available. Select a city to view insights.',
        'pricePerSqftChart': 'Price per sqft data unavailable for current filters.'
    };
    
    const message = messages[canvasId] || 'Apply filters or select a city to view insights.';
    
    const parent = canvas.parentElement;
    const noDataDiv = document.createElement('div');
    noDataDiv.className = 'flex items-center justify-center h-full text-slate-500';
    noDataDiv.innerHTML = '<div class="text-center"><i class="fas fa-chart-bar text-4xl mb-2 opacity-50"></i><p class="text-sm">' + message + '</p></div>';
    canvas.style.display = 'none';
    parent.appendChild(noDataDiv);
}

/**
 * Check if array has valid data
 * Returns true only if array exists, is non-empty, and has at least one valid value
 */
function hasData(arr) {
    if (!arr || !Array.isArray(arr) || arr.length === 0) return false;
    // Check if at least one value is valid (not null, undefined, or NaN)
    return arr.some(v => v !== null && v !== undefined && !Number.isNaN(v));
}

/**
 * Sanitize numeric values - replace NaN/null with 0
 * Prevents chart rendering issues from bad data
 */
function sanitizeValues(arr) {
    if (!arr || !Array.isArray(arr)) return [];
    return arr.map(v => {
        if (v === null || v === undefined || Number.isNaN(v)) return 0;
        return typeof v === 'number' ? v : parseFloat(v) || 0;
    });
}

/**
 * Initialize ROI Comparison Bar Chart
 * Shows top 10 properties by Buy Advantage (wealth_buying vs wealth_renting)
 * Positive % = buying is advantageous, Negative % = renting is better
 */
function initROIChart() {
    const ctx = document.getElementById('roiChart');
    if (!ctx) return;

    // Check for data
    if (!hasData(chartData.roi_labels) || !hasData(chartData.roi_values)) {
        showNoDataMessage('roiChart');
        return;
    }

    // Determine bar colors: green for positive (buy advantage), red for negative (rent better)
    const barColors = chartData.roi_values.map(value => 
        value >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
    );
    const borderColors = chartData.roi_values.map(value => 
        value >= 0 ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)'
    );

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.roi_labels,
            datasets: [{
                label: 'Buy Advantage (%)',
                data: chartData.roi_values,
                backgroundColor: barColors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 30
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            const label = value >= 0 ? 'Buy Advantage: +' : 'Rent Better: ';
                            return label + value.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    // Allow negative values since some properties favor renting
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        },
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        font: {
                            size: 10
                        },
                        color: '#94a3b8'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

/**
 * Initialize Buy vs Rent Doughnut Chart
 * Shows distribution of buy vs rent recommendations
 */
function initBuyVsRentChart() {
    const ctx = document.getElementById('buyVsRentChart');
    if (!ctx) return;

    // Check for data - buy_vs_rent_values should have non-zero values
    const hasValidData = chartData.buy_vs_rent_values && 
                         chartData.buy_vs_rent_values.some(v => v > 0);
    if (!hasValidData) {
        showNoDataMessage('buyVsRentChart');
        return;
    }

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartData.buy_vs_rent_labels,
            datasets: [{
                label: 'Properties',
                data: chartData.buy_vs_rent_values,
                backgroundColor: [
                    'rgba(34, 197, 94, 0.8)',   // Green for Buy
                    'rgba(99, 102, 241, 0.8)'   // Indigo for Rent
                ],
                borderColor: [
                    'rgba(34, 197, 94, 1)',
                    'rgba(99, 102, 241, 1)'
                ],
                borderWidth: 3,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: {
                            size: 13,
                            weight: '500'
                        },
                        color: '#e2e8f0',
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return label + ': ' + value + ' (' + percentage + '%)';
                        }
                    }
                }
            },
            cutout: '65%',
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

/**
 * Initialize Price Distribution Bar Chart
 * Shows average property prices by city
 */
function initPriceDistributionChart() {
    const ctx = document.getElementById('priceDistributionChart');
    if (!ctx) return;

    if (!hasData(chartData.price_distribution_labels)) {
        showNoDataMessage('priceDistributionChart');
        return;
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.price_distribution_labels,
            datasets: [{
                label: 'Average Price (₹)',
                data: chartData.price_distribution_values,
                backgroundColor: createGradient(ctx, 'rgba(16, 185, 129, 0.8)', 'rgba(5, 150, 105, 0.8)'),
                borderColor: 'rgba(16, 185, 129, 1)',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',  // Horizontal bar chart
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.x;
                            return 'Avg Price: ₹' + formatIndianCurrency(value);
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '₹' + formatShortCurrency(value);
                        },
                        font: {
                            size: 11
                        },
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                y: {
                    ticks: {
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        color: '#e2e8f0'
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

/**
 * Create gradient for chart backgrounds
 * @param {HTMLCanvasElement} canvasEl - The canvas element
 * @param {string} color1 - Start color
 * @param {string} color2 - End color
 * @returns {CanvasGradient|string} - Gradient or fallback color
 */
function createGradient(canvasEl, color1, color2) {
    try {
        const ctx = canvasEl.getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, canvasEl.height || 300);
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    } catch (e) {
        console.warn('Gradient creation failed, using fallback color:', e);
        return color1; // Fallback to solid color
    }
}

/**
 * Initialize City Properties Count Chart
 * Shows number of properties per city
 */
function initCityPropertiesChart() {
    const ctx = document.getElementById('cityPropertiesChart');
    if (!ctx) return;

    // Guard: Check for valid data before initializing chart
    if (!hasData(chartData.city_properties_labels) || !hasData(chartData.city_properties_values)) {
        showNoDataMessage('cityPropertiesChart');
        return;
    }

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartData.city_properties_labels,
            datasets: [{
                label: 'Properties',
                data: chartData.city_properties_values,
                backgroundColor: [
                    'rgba(139, 92, 246, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(236, 72, 153, 0.8)',
                    'rgba(99, 102, 241, 0.8)',
                    'rgba(20, 184, 166, 0.8)'
                ],
                borderColor: '#1e293b',
                borderWidth: 3,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        color: '#e2e8f0',
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                        }
                    }
                }
            },
            cutout: '60%',
            animation: {
                animateRotate: true,
                duration: 1500
            }
        }
    });
}

/**
 * Initialize Location Price Chart
 * Shows top locations by average price
 */
function initLocationPriceChart() {
    const ctx = document.getElementById('locationPriceChart');
    if (!ctx) return;

    // Guard: Check for valid data before initializing chart
    if (!hasData(chartData.location_price_labels) || !hasData(chartData.location_price_values)) {
        showNoDataMessage('locationPriceChart');
        return;
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.location_price_labels,
            datasets: [{
                label: 'Average Price',
                data: chartData.location_price_values,
                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                borderColor: 'rgba(239, 68, 68, 1)',
                borderWidth: 2,
                borderRadius: 6,
                barThickness: 25
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return 'Avg Price: ₹' + formatIndianCurrency(context.parsed.x);
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '₹' + formatShortCurrency(value);
                        },
                        font: { size: 10 },
                        color: '#94a3b8'
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    ticks: {
                        font: { size: 10, weight: '500' },
                        color: '#e2e8f0'
                    },
                    grid: { display: false }
                }
            },
            animation: { duration: 1500 }
        }
    });
}

/**
 * Initialize Price Per Sqft Chart
 * Shows average price per sqft by city
 */
function initPricePerSqftChart() {
    const ctx = document.getElementById('pricePerSqftChart');
    if (!ctx) return;

    // Guard: Check for valid data before initializing chart
    if (!hasData(chartData.price_per_sqft_labels) || !hasData(chartData.price_per_sqft_values)) {
        showNoDataMessage('pricePerSqftChart');
        return;
    }

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.price_per_sqft_labels,
            datasets: [{
                label: 'Price per Sqft',
                data: chartData.price_per_sqft_values,
                backgroundColor: createGradient(ctx, 'rgba(245, 158, 11, 0.8)', 'rgba(217, 119, 6, 0.8)'),
                borderColor: 'rgba(245, 158, 11, 1)',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return 'Price/sqft: ₹' + context.parsed.y.toLocaleString('en-IN');
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '₹' + value.toLocaleString('en-IN');
                        },
                        font: { size: 11 },
                        color: '#94a3b8'
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: {
                        font: { size: 11 },
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: { display: false }
                }
            },
            animation: { duration: 1500 }
        }
    });
}

/**
 * Format currency in Indian numbering system (Lakhs/Crores)
 */
function formatIndianCurrency(amount) {
    if (amount >= 10000000) {
        return (amount / 10000000).toFixed(2) + ' Cr';
    } else if (amount >= 100000) {
        return (amount / 100000).toFixed(2) + ' L';
    } else if (amount >= 1000) {
        return (amount / 1000).toFixed(0) + ' K';
    } else {
        return amount.toFixed(0);
    }
}

/**
 * Format currency for axis labels (shorter version)
 */
function formatShortCurrency(amount) {
    if (amount >= 10000000) {
        return (amount / 10000000).toFixed(1) + 'Cr';
    } else if (amount >= 100000) {
        return (amount / 100000).toFixed(0) + 'L';
    } else {
        return (amount / 1000).toFixed(0) + 'K';
    }
}

/**
 * Add smooth animation to number counters on KPI cards
 */
function animateValue(element, start, end, duration) {
    if (!element) return;
    
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;
    
    const timer = setInterval(function() {
        current += increment;
        element.textContent = current.toLocaleString('en-IN');
        
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
}

// Optional: Add animation to KPI numbers on page load
window.addEventListener('load', function() {
    const kpiElements = document.querySelectorAll('[data-animate-value]');
    kpiElements.forEach(function(element) {
        const targetValue = parseInt(element.dataset.animateValue);
        animateValue(element, 0, targetValue, 1000);
    });
});
