# WiBaSL Dataset Overview

 

The data records cited in the **WiBaSL** article are stored [here](https://github.com/cpi-lab-buet/WiBaSL/tree/main/WiBaSL/data). The dataset is organized in a main folder, which is subdivided into individual folders corresponding to each volunteer. 

 

Table [Table 1](#) in the paper (Table~\ref{tab:volunteers}) provides relevant details about the participating volunteers.

 

---

 

## Folder Structure

 

Each volunteer folder contains **480 CSI recordings**, comprising:

- **20 samples** for each of the **24 Bangladeshi Sign Language (BaSL)** signs.

 

Each recording is stored as a `.dat` file.  

The filename encodes:

- The **volunteer serial**

- The **corresponding activity type**  

This allows for easy identification of each sample.

 

---

 

## CSI File Information

 

- Collected using: **Intel 5300 CSI Tool**

- Parsing: Compatible with **CSI-Kit utilities** [[CSI-Kit Reference](https://github.com/gforbes/csi-kit)].

 

### Metadata Extractable:

- Chipset used  

- Backend type  

- Channel bandwidth  

- Antenna configuration  

- Frame count  

- Subcarrier count  

- Recording duration  

- Average sampling rate  

- Average RSSI  

- CSI tensor shape

 

---

 

## Data Format

 

Each `.dat` file contains:

- Approx. **10 seconds** of CSI data for a specific **hand gesture**

- Sampled at **20 packets per second**

- Structured in **4D format**:  

(frames, subcarriers, transmit antennas, receive antennas)

---
