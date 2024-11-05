import numpy as np
import mne
from ripser import Rips
from persim import plot_diagrams
from gudhi.representations import Landscape
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# File path template for participants
file_template = "/content/sub-{:02d}_ses-1_task-fuzzysemanticrecognition_eeg.edf"

# List to store H1 persistence landscapes for each participant
h1_landscapes = []

# Function to load, preprocess, and embed EEG data for a single channel
def process_participant(file_path, channel_name="Fz"):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_channels([channel_name])
    raw.filter(1., 40., fir_design='firwin')  # Bandpass filter
    data, times = raw[:]
    
    # Segment the data around a sample event (assuming 10s start here as an example)
    event_start = 10.0  # seconds
    sampling_rate = len(data.flatten()) / times[-1]
    start_idx = int((event_start - 0.5) * sampling_rate)  # 500 ms before
    end_idx = int((event_start + 1.0) * sampling_rate)    # 1000 ms after
    segmented_data = data.flatten()[start_idx:end_idx]
    
    # Time Delay Embedding
    delay = 5
    embedding_dim = 10
    embedded_data = np.array([
        segmented_data[i:i + embedding_dim * delay:delay]
        for i in range(len(segmented_data) - (embedding_dim - 1) * delay)
    ])
    
    # Standardize embedded data
    scaler = StandardScaler()
    embedded_data = scaler.fit_transform(embedded_data)
    
    # Persistent Homology and Persistence Diagram (specifically for H1)
    rips = Rips()
    diagrams = rips.fit_transform(embedded_data)
    
    # Persistence Landscape for H1
    landscape = Landscape()
    h1_landscapes = landscape.fit_transform([diagrams[1]])  # For H1 features
    
    return diagrams, h1_landscapes[0]

# Process each participant and store their H1 landscapes
for i in range(1, 6):  # Participant numbers from 1 to 30
    file_path = file_template.format(i)
    diagrams, h1_landscape = process_participant(file_path)
    h1_landscapes.append(h1_landscape)

# Plot Average Persistence Landscape for H1 across all participants
average_h1_landscape = np.mean(np.array(h1_landscapes), axis=0)

plt.figure(figsize=(10, 6))
for i, line in enumerate(average_h1_landscape[:5]):  # Plot first few layers of the average landscape
    plt.plot(line, label=f"Layer {i+1}")
plt.xlabel("Time")
plt.ylabel("Average Landscape Value")
plt.legend()
plt.title("Average Persistence Landscape for H1 Across Participants")
plt.show()

# Example: Plot individual H1 persistence landscapes for first few participants
for i in range(5):  # Plot for first 5 participants
    plt.figure(figsize=(10, 6))
    for layer in h1_landscapes[i][:5]:  # Plot first few layers of each individual landscape
        plt.plot(layer)
    plt.title(f"H1 Persistence Landscape for Participant {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Landscape Value")
    plt.show()
