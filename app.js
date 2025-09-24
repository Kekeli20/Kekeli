class HypoxiaMonitorApp {
    constructor() {
        this.bluetoothManager = new BluetoothManager();
        this.chartManager = new ChartManager();
        this.dataStorage = [];
        this.sessionStartTime = null;
        this.sessionTimer = null;
        this.currentSpo2 = null;
        this.currentHR = null;
        this.hypoxiaAlertShown = false;
        
        this.initializeUI();
        this.setupEventListeners();
        
        // Check for Web Bluetooth support
        if (!navigator.bluetooth) {
            this.showError('Web Bluetooth is not supported in this browser. Please use Chrome, Edge, or another compatible browser.');
        }
    }

    initializeUI() {
        this.elements = {
            connectBtn: document.getElementById('connectBtn'),
            disconnectBtn: document.getElementById('disconnectBtn'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            spo2Value: document.getElementById('spo2Value'),
            spo2Status: document.getElementById('spo2Status'),
            hrValue: document.getElementById('hrValue'),
            hrStatus: document.getElementById('hrStatus'),
            clearDataBtn: document.getElementById('clearDataBtn'),
            pauseBtn: document.getElementById('pauseBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            clearStorageBtn: document.getElementById('clearStorageBtn'),
            totalRecords: document.getElementById('totalRecords'),
            sessionDuration: document.getElementById('sessionDuration'),
            recentDataList: document.getElementById('recentDataList'),
            alertModal: document.getElementById('alertModal'),
            acknowledgeAlert: document.getElementById('acknowledgeAlert'),
            errorModal: document.getElementById('errorModal'),
            closeError: document.getElementById('closeError'),
            errorMessage: document.getElementById('errorMessage')
        };
    }

    setupEventListeners() {
        // Bluetooth connection
        this.elements.connectBtn.addEventListener('click', () => this.connectDevice());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnectDevice());
        
        // Chart controls
        this.elements.clearDataBtn.addEventListener('click', () => this.clearChartData());
        this.elements.pauseBtn.addEventListener('click', () => this.togglePause());
        
        // Data management
        this.elements.downloadBtn.addEventListener('click', () => this.downloadData());
        this.elements.clearStorageBtn.addEventListener('click', () => this.clearAllData());
        
        // Modal controls
        this.elements.acknowledgeAlert.addEventListener('click', () => this.acknowledgeHypoxiaAlert());
        this.elements.closeError.addEventListener('click', () => this.closeErrorModal());
        
        // Set up Bluetooth data callback
        this.bluetoothManager.setDataCallback((type, value) => this.handleBluetoothData(type, value));
    }

    async connectDevice() {
        try {
            this.elements.connectBtn.disabled = true;
            this.elements.connectBtn.innerHTML = '<span class="loading"></span> Connecting...';
            
            await this.bluetoothManager.connect();
            
            this.updateConnectionStatus(true);
            this.startSession();
            
        } catch (error) {
            console.error('Connection failed:', error);
            
            // Check if user cancelled the device selection
            if (error.name === 'NotFoundError' || error.name === 'AbortError') {
                console.log('User cancelled device selection');
            } else {
                this.showError(`Failed to connect: ${error.message}`);
            }
            this.updateConnectionStatus(false);
        } finally {
            this.elements.connectBtn.disabled = false;
            this.elements.connectBtn.innerHTML = '<span class="btn-icon">🔵</span> Connect to Device';
        }
    }

    async disconnectDevice() {
        try {
            await this.bluetoothManager.disconnect();
            this.updateConnectionStatus(false);
            this.stopSession();
        } catch (error) {
            console.error('Disconnect failed:', error);
        }
    }

    handleBluetoothData(type, value) {
        const timestamp = new Date();
        
        switch (type) {
            case 'spo2':
                this.updateSpo2(value, timestamp);
                break;
            case 'heartRate':
                this.updateHeartRate(value, timestamp);
                break;
            case 'disconnected':
                this.updateConnectionStatus(false);
                this.stopSession();
                break;
        }
    }

    updateSpo2(value, timestamp) {
        this.currentSpo2 = Math.round(value);
        this.elements.spo2Value.textContent = this.currentSpo2;
        
        // Update status based on SpO2 value
        const spo2Card = document.querySelector('.spo2-card');
        if (this.currentSpo2 < 90) {
            this.elements.spo2Status.textContent = 'Critical';
            this.elements.spo2Status.className = 'vital-status critical';
            spo2Card.classList.add('critical');
            this.checkHypoxiaAlert();
        } else if (this.currentSpo2 < 95) {
            this.elements.spo2Status.textContent = 'Warning';
            this.elements.spo2Status.className = 'vital-status warning';
            spo2Card.classList.remove('critical');
        } else {
            this.elements.spo2Status.textContent = 'Normal';
            this.elements.spo2Status.className = 'vital-status normal';
            spo2Card.classList.remove('critical');
            this.hypoxiaAlertShown = false; // Reset alert for next occurrence
        }

        // Add to chart
        this.chartManager.addDataPoint(this.currentSpo2);
        
        // Store data
        this.storeDataPoint('spo2', this.currentSpo2, timestamp);
    }

    updateHeartRate(value, timestamp) {
        this.currentHR = Math.round(value);
        this.elements.hrValue.textContent = this.currentHR;
        
        // Update status based on heart rate
        if (this.currentHR < 60 || this.currentHR > 100) {
            this.elements.hrStatus.textContent = 'Warning';
            this.elements.hrStatus.className = 'vital-status warning';
        } else {
            this.elements.hrStatus.textContent = 'Normal';
            this.elements.hrStatus.className = 'vital-status normal';
        }
        
        // Store data
        this.storeDataPoint('heartRate', this.currentHR, timestamp);
    }

    storeDataPoint(type, value, timestamp) {
        this.dataStorage.push({
            timestamp: timestamp,
            type: type,
            value: value
        });
        
        this.updateDataStats();
        this.updateRecentDataDisplay();
    }

    updateDataStats() {
        this.elements.totalRecords.textContent = this.dataStorage.length;
    }

    updateRecentDataDisplay() {
        const recentData = this.dataStorage.slice(-10).reverse();
        
        if (recentData.length === 0) {
            this.elements.recentDataList.innerHTML = '<p class="no-data">No data available</p>';
            return;
        }
        
        const html = recentData.map(entry => {
            const time = entry.timestamp.toLocaleTimeString();
            const typeDisplay = entry.type === 'spo2' ? 'SpO₂' : 'HR';
            const unit = entry.type === 'spo2' ? '%' : 'BPM';
            
            return `
                <div class="data-entry">
                    <span>${time} - ${typeDisplay}</span>
                    <span>${entry.value}${unit}</span>
                </div>
            `;
        }).join('');
        
        this.elements.recentDataList.innerHTML = html;
    }

    checkHypoxiaAlert() {
        if (this.currentSpo2 < 90 && !this.hypoxiaAlertShown) {
            this.showHypoxiaAlert();
            this.hypoxiaAlertShown = true;
        }
    }

    showHypoxiaAlert() {
        this.elements.alertModal.classList.add('show');
        
        // Play alert sound (if supported)
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 1);
        } catch (error) {
            console.log('Audio alert not available');
        }
    }

    acknowledgeHypoxiaAlert() {
        this.elements.alertModal.classList.remove('show');
    }

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.classList.add('show');
    }

    closeErrorModal() {
        this.elements.errorModal.classList.remove('show');
    }

    updateConnectionStatus(connected) {
        if (connected) {
            this.elements.statusIndicator.classList.add('connected');
            this.elements.statusText.textContent = 'Connected';
            this.elements.connectBtn.style.display = 'none';
            this.elements.disconnectBtn.style.display = 'inline-flex';
            this.elements.disconnectBtn.disabled = false;
        } else {
            this.elements.statusIndicator.classList.remove('connected');
            this.elements.statusText.textContent = 'Disconnected';
            this.elements.connectBtn.style.display = 'inline-flex';
            this.elements.disconnectBtn.style.display = 'none';
            this.elements.disconnectBtn.disabled = true;
            
            // Reset vital signs display
            this.elements.spo2Value.textContent = '--';
            this.elements.hrValue.textContent = '--';
            this.elements.spo2Status.textContent = '';
            this.elements.spo2Status.className = 'vital-status';
            this.elements.hrStatus.textContent = '';
            this.elements.hrStatus.className = 'vital-status';
            
            document.querySelector('.spo2-card').classList.remove('critical');
        }
    }

    startSession() {
        this.sessionStartTime = new Date();
        this.sessionTimer = setInterval(() => {
            this.updateSessionDuration();
        }, 1000);
    }

    stopSession() {
        if (this.sessionTimer) {
            clearInterval(this.sessionTimer);
            this.sessionTimer = null;
        }
        this.elements.sessionDuration.textContent = '00:00:00';
    }

    updateSessionDuration() {
        if (!this.sessionStartTime) return;
        
        const now = new Date();
        const duration = Math.floor((now - this.sessionStartTime) / 1000);
        
        const hours = Math.floor(duration / 3600);
        const minutes = Math.floor((duration % 3600) / 60);
        const seconds = duration % 60;
        
        this.elements.sessionDuration.textContent = 
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    clearChartData() {
        this.chartManager.clearData();
    }

    togglePause() {
        const isPaused = this.chartManager.togglePause();
        this.elements.pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
    }

    downloadData() {
        if (this.dataStorage.length === 0) {
            this.showError('No data available to download.');
            return;
        }
        
        const csvContent = this.generateCSV();
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `hypoxia-monitor-data-${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    generateCSV() {
        const headers = ['Timestamp', 'Date', 'Time', 'Type', 'Value', 'Unit', 'Status'];
        const rows = [headers];
        
        this.dataStorage.forEach(entry => {
            const date = entry.timestamp.toLocaleDateString();
            const time = entry.timestamp.toLocaleTimeString();
            const type = entry.type === 'spo2' ? 'SpO2' : 'Heart Rate';
            const unit = entry.type === 'spo2' ? '%' : 'BPM';
            
            let status = 'Normal';
            if (entry.type === 'spo2') {
                if (entry.value < 90) status = 'Severe Hypoxia';
                else if (entry.value < 95) status = 'Mild Hypoxia';
            } else {
                if (entry.value < 60 || entry.value > 100) status = 'Abnormal';
            }
            
            rows.push([
                entry.timestamp.toISOString(),
                date,
                time,
                type,
                entry.value,
                unit,
                status
            ]);
        });
        
        return rows.map(row => row.join(',')).join('\n');
    }

    clearAllData() {
        if (confirm('Are you sure you want to clear all recorded data? This action cannot be undone.')) {
            this.dataStorage = [];
            this.chartManager.clearData();
            this.updateDataStats();
            this.updateRecentDataDisplay();
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.hypoxiaApp = new HypoxiaMonitorAppWithAI();
});

