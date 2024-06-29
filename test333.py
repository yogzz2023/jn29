import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6,1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3,1))
        self.first_measurement = True
        self.prev_r = 0
        self.prev_az = 0
        self.prev_el = 0
        self.chi2_threshold = 5.99  # Chi-square threshold for selection (adjust as needed)

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)
        
    def initialize_measurement_for_filtering(self, x, y, z, vx, vy, vz, mt):
        self.Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def update_step(self, grouped_measurements, current_time):
        dt = current_time - self.Meas_Time
        
        for group in grouped_measurements:
            # Check chi-square test for group selection
            if self.perform_chi_square_test(group):
                selected_group = group
                break
        else:
            selected_group = grouped_measurements[-1]  # Use last group if none satisfy

        # Use first measurement of selected group for update
        r, az, el, mt = selected_group[0]
        self.Z = np.array([[r], [az], [el]])
        
        if self.Meas_Time == 0:
            self.initialize_filter_state(r, az, el, 0, 0, 0, mt)
        else:
            if self.first_measurement:
                self.prev_r, self.prev_az, self.prev_el = r, az, el
                self.first_measurement = False
            else:
                vx = (r - self.prev_r) / dt
                vy = (az - self.prev_az) / dt
                vz = (el - self.prev_el) / dt
                self.initialize_filter_state(r, az, el, vx, vy, vz, mt)

        # Process remaining measurements in the selected group
        for r, az, el, mt in selected_group[1:]:
            self.predict_step(mt)
            self.update_filter_state(r, az, el, mt)

    def perform_chi_square_test(self, group):
        # Calculate predicted measurement based on current filter state
        predicted_measurement = np.dot(self.H, self.Sf).reshape(-1)
        
        # Calculate residual
        residuals = []
        for measurement in group:
            r, az, el, _ = measurement
            actual_measurement = np.array([r, az, el])
            residual = actual_measurement - predicted_measurement[:3]  
            residuals.append(residual)
        
        cov_inv = np.linalg.inv(self.R)
        
        # Calculate chi-square statistic
        chi2_values = []
        for residual in residuals:
            chi2_value = np.dot(residual.T, np.dot(cov_inv, residual))
            chi2_values.append(chi2_value)
        
        # Compare with threshold
        min_chi2 = min(chi2_values)
        return min_chi2 < self.chi2_threshold

    def update_filter_state(self, r, az, el, mt):
        Inn = self.Z - np.dot(self.H, self.Sf)
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def group_measurements(self, measurements, max_time_diff):
        grouped_measurements = []
        current_group = []
        for i in range(len(measurements)):
            if i == 0 or measurements[i][3] - measurements[i-1][3] < max_time_diff:
                current_group.append(measurements[i])
            else:
                if current_group:
                    grouped_measurements.append(current_group)
                current_group = [measurements[i]]
        if current_group:
            grouped_measurements.append(current_group)
        return grouped_measurements

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z/np.sqrt(x**2 + y**2))*180/3.14
    az = math.atan(y/x)

    if x > 0.0:
        az = 3.14/2 - az
    else:
        az = 3*3.14/2 - az

    az = az*180/3.14

    if(az < 0.0):
        az = (360 + az)

    if(az > 360):
        az = (az - 360)

    return r, az, el

def cart2sph2(x:float, y:float, z:float, filtered_values_csv):
    r=[]
    az=[]
    el=[]
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))

        if x[i] > 0.0:
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]

        az[i] = az[i]*180/3.14

        if(az[i] < 0.0):
            az[i] = (360 + az[i])

        if(az[i] > 360):
            az[i] = (az[i] - 360)

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

# Main code
kalman_filter = CVFilter()

csv_file_path = 'ttk_52_test.csv'
measurements = read_measurements_from_csv(csv_file_path)
grouped_measurements = kalman_filter.group_measurements(measurements, max_time_diff=50)

csv_file_predicted = "ttk_52_test.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['FT', 'FX', 'FY', 'FZ']].values
measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values


A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
# number = 1000
# result = np.divide(A[0], number)

time_list = []
r_list = []
az_list = []
el_list = []

for i, (r, az, el, mt) in enumerate(measurements):
    if i == 0:
        kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
    elif i == 1:
        prev_r, prev_az, prev_el = measurements[i-1][:3]
        dt = mt - measurements[i-1][3]
        vx = (r - prev_r) / dt
        vy = (az - prev_az) / dt
        vz = (el - prev_el) / dt
        print("vz",vz)
        kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
    else:
        kalman_filter.update_step(grouped_measurements[:i+1], mt)
        time_list.append(mt)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, r_list, label='filtered range (code)', color='green', marker='o')
plt.scatter(filtered_values_csv[:, 0], A[0], label='filtered range (track id 31)', color='red', marker='*')
plt.scatter(measured_values_csv[:, 0],measured_values_csv[:, 1], label='filtered range (track id 31)', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Range (r)', color='black')
plt.title('Range vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, az_list, label='filtered azimuth (code)', color='green', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.scatter(measured_values_csv[:, 0],measured_values_csv[:, 2], label='filtered az (track id 31)', color='blue', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Azimuth (az)', color='black')
plt.title('Azimuth vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(facecolor="white")
plt.scatter(time_list, el_list, label='filtered elevation (code)', color='green', marker='*')
plt.scatter(measured_values_csv[:, 0],measured_values_csv[:, 3], label='filtered el (track id 31)', color='blue', marker='*')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time', color='black')
plt.ylabel('Elevation (el)', color='black')
plt.title('Elevation vs. Time', color='black')
plt.grid(color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()
