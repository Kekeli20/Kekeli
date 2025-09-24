class ChartManager {
    constructor() {
        this.chart = null;
        this.ctx = document.getElementById('spo2Chart').getContext('2d');
        this.maxDataPoints = 50;
        this.isPaused = false;
        this.hypoxiaThreshold = 90;
        this.mildHypoxiaThreshold = 95;
        
        this.initChart();
    }

    initChart() {
        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'SpO₂ (%)',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300,
                    easing: 'easeInOutQuad'
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'SpO₂ (%)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        min: 50,
                        max: 100,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            stepSize: 5
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Time: ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                const value = context.parsed.y;
                                let status = 'Normal';
                                if (value < 90) status = 'Severe Hypoxia';
                                else if (value < 95) status = 'Mild Hypoxia';
                                return [`SpO₂: ${value}%`, `Status: ${status}`];
                            }
                        }
                    }
                },
                elements: {
                    point: {
                        pointStyle: function(context) {
                            if (!context.parsed) return 'circle';
                            const value = context.parsed.y;
                            if (value < 90) {
                                return 'circle';
                            }
                            return 'circle';
                        },
                        backgroundColor: function(context) {
                            if (!context.parsed) return '#10b981';
                            const value = context.parsed.y;
                            if (value < 90) return '#ef4444';
                            if (value < 95) return '#f59e0b';
                            return '#10b981';
                        },
                        borderColor: function(context) {
                            if (!context.parsed) return '#059669';
                            const value = context.parsed.y;
                            if (value < 90) return '#dc2626';
                            if (value < 95) return '#d97706';
                            return '#059669';
                        }
                    }
                }
            }
        });

        // Add horizontal reference lines for hypoxia thresholds
        this.addReferenceLines();
    }

    addReferenceLines() {
        const chartArea = this.chart.chartArea;
        const ctx = this.chart.ctx;
        
        // Add plugin to draw reference lines
        Chart.register({
            id: 'referenceLines',
            afterDraw: (chart) => {
                if (chart.canvas.id !== 'spo2Chart') return;
                
                const ctx = chart.ctx;
                const chartArea = chart.chartArea;
                const yScale = chart.scales.y;
                
                ctx.save();
                
                // Severe hypoxia line (90%)
                const severeY = yScale.getPixelForValue(90);
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(chartArea.left, severeY);
                ctx.lineTo(chartArea.right, severeY);
                ctx.stroke();
                
                // Mild hypoxia line (95%)
                const mildY = yScale.getPixelForValue(95);
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(chartArea.left, mildY);
                ctx.lineTo(chartArea.right, mildY);
                ctx.stroke();
                
                ctx.restore();
            }
        });
    }

    addDataPoint(spo2Value) {
        if (this.isPaused) return;

        const now = new Date();
        const timeLabel = now.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        this.chart.data.labels.push(timeLabel);
        this.chart.data.datasets[0].data.push(spo2Value);

        // Remove old data points if we exceed the maximum
        if (this.chart.data.labels.length > this.maxDataPoints) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
        }

        // Update point colors based on hypoxia status
        this.updatePointColors();
        
        this.chart.update('none');
    }

    updatePointColors() {
        const dataset = this.chart.data.datasets[0];
        const data = dataset.data;
        
        // Create point colors array if it doesn't exist
        if (!dataset.pointBackgroundColor || !Array.isArray(dataset.pointBackgroundColor)) {
            dataset.pointBackgroundColor = [];
            dataset.pointBorderColor = [];
        }

        // Update colors for all points
        for (let i = 0; i < data.length; i++) {
            const value = data[i];
            if (value < 90) {
                dataset.pointBackgroundColor[i] = '#ef4444';
                dataset.pointBorderColor[i] = '#dc2626';
            } else if (value < 95) {
                dataset.pointBackgroundColor[i] = '#f59e0b';
                dataset.pointBorderColor[i] = '#d97706';
            } else {
                dataset.pointBackgroundColor[i] = '#10b981';
                dataset.pointBorderColor[i] = '#059669';
            }
        }
    }

    clearData() {
        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.data.datasets[0].pointBackgroundColor = [];
        this.chart.data.datasets[0].pointBorderColor = [];
        this.chart.update();
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        return this.isPaused;
    }

    isPausedState() {
        return this.isPaused;
    }

    getAllData() {
        const labels = this.chart.data.labels;
        const data = this.chart.data.datasets[0].data;
        
        return labels.map((label, index) => ({
            time: label,
            spo2: data[index]
        }));
    }
}

// Make ChartManager globally available
window.ChartManager = ChartManager;