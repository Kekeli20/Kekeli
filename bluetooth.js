// Define UUIDs
const CUSTOM_SERVICE_UUID       = "91bad492-b950-4226-aa2b-4ede9fa42f59";  // 👈 Service
const HEART_RATE_CHAR_UUID      = "ca73b3ba-39f6-4ab3-91ae-186dc9577d99";
const SPO2_CHAR_UUID            = "cba1d466-344c-4be3-ab3f-189f80dd7518";
const SIGNAL_QUALITY_CHAR_UUID  = "87654321-4321-4321-4321-cba987654321";

class BluetoothManager {
    constructor() {
        this.device = null;
        this.server = null;
        this.service = null;
        this.spo2Characteristic = null;
        this.hrCharacteristic = null;
        this.signalQualityCharacteristic = null;
        this.isConnected = false;
        this.dataCallback = null;
    }

    async connect() {
        try {
            console.log('Requesting Bluetooth device...');
            
            this.device = await navigator.bluetooth.requestDevice({
                filters: [
                    { name: 'ESP32' },
                    { namePrefix: 'ESP32' },
                    { namePrefix: 'HypoxGuard' },
                    { namePrefix: 'SpO2' }
                ],
                optionalServices: [CUSTOM_SERVICE_UUID, 'device_information'],
                acceptAllDevices: false
            });

            console.log('Selected device:', this.device.name);
            
            this.device.addEventListener('gattserverdisconnected', () => {
                console.log('Device disconnected');
                this.onDisconnected();
            });

            console.log('Connecting to GATT Server...');
            this.server = await this.device.gatt.connect();

            console.log('Getting service...');
            this.service = await this.server.getPrimaryService(CUSTOM_SERVICE_UUID);

            console.log('Getting characteristics...');
            this.hrCharacteristic = await this.service.getCharacteristic(HEART_RATE_CHAR_UUID);
            this.spo2Characteristic = await this.service.getCharacteristic(SPO2_CHAR_UUID);

            // Optional extra char
            try {
                this.signalQualityCharacteristic = await this.service.getCharacteristic(SIGNAL_QUALITY_CHAR_UUID);
                console.log('Found Signal Quality characteristic');
            } catch (err) {
                console.log('Signal Quality characteristic not found (optional)');
            }

            // Start notifications
            console.log('Starting notifications...');
            await this.spo2Characteristic.startNotifications();
            this.spo2Characteristic.addEventListener('characteristicvaluechanged', (event) => {
                this.handleSpo2Data(event);
            });

            await this.hrCharacteristic.startNotifications();
            this.hrCharacteristic.addEventListener('characteristicvaluechanged', (event) => {
                this.handleHrData(event);
            });

            this.isConnected = true;
            console.log('Successfully connected and configured notifications ✅');

            return true;
        } catch (error) {
            console.error('Connection error:', error);
            this.cleanup();
            throw error;
        }
    }

    handleSpo2Data(event) {
        const value = event.target.value;
        let spo2;

        if (value.byteLength >= 4) {
            spo2 = value.getFloat32(0, true);
        } else if (value.byteLength >= 2) {
            spo2 = value.getUint16(0, true);
        } else {
            spo2 = value.getUint8(0);
        }

        if (spo2 >= 0 && spo2 <= 100) {
            if (this.dataCallback) {
                this.dataCallback('spo2', spo2);
            }
        }
    }

    handleHrData(event) {
        const value = event.target.value;
        let heartRate;

        if (value.byteLength >= 4) {
            heartRate = value.getFloat32(0, true);
        } else if (value.byteLength >= 2) {
            heartRate = value.getUint16(0, true);
        } else {
            heartRate = value.getUint8(0);
        }

        if (heartRate >= 0 && heartRate <= 200) {
            if (this.dataCallback) {
                this.dataCallback('heartRate', heartRate);
            }
        }
    }

    async disconnect() {
        try {
            if (this.spo2Characteristic) {
                await this.spo2Characteristic.stopNotifications();
            }
            if (this.hrCharacteristic) {
                await this.hrCharacteristic.stopNotifications();
            }
            if (this.server) {
                await this.server.disconnect();
            }
        } catch (error) {
            console.error('Disconnect error:', error);
        } finally {
            this.cleanup();
        }
    }

    cleanup() {
        this.device = null;
        this.server = null;
        this.service = null;
        this.spo2Characteristic = null;
        this.hrCharacteristic = null;
        this.signalQualityCharacteristic = null;
        this.isConnected = false;
    }

    onDisconnected() {
        this.cleanup();
        if (this.dataCallback) {
            this.dataCallback('disconnected', null);
        }
    }

    setDataCallback(callback) {
        this.dataCallback = callback;
    }

    getConnectionInfo() {
        return {
            deviceName: this.device ? this.device.name : null,
            serviceUUID: CUSTOM_SERVICE_UUID,
            spo2CharacteristicUUID: SPO2_CHAR_UUID,
            hrCharacteristicUUID: HEART_RATE_CHAR_UUID,
            signalQualityCharacteristicUUID: SIGNAL_QUALITY_CHAR_UUID,
            isConnected: this.isConnected
        };
    }
}

// Make BluetoothManager globally available
window.BluetoothManager = BluetoothManager;
