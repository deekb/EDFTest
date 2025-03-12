# Signal Filtering and Analysis from EDF Files (Motor Coil Signals)

This Python project demonstrates how to process, filter, and analyze motor coil data stored in an EDF (European Data Format) file. The main features of this project include:

- Reading an EDF file containing motor coil data.
- Applying a low-pass Butterworth filter to the raw signal.
- Comparing the filtered signal to an optimal sine wave based on the dominant frequency.
- Visualizing the results, including relevant statistics like RMS and average.

## Features

- **Load EDF Data**: The project loads motor coil data from an EDF file.
- **Low-pass Filtering**: A Butterworth low-pass filter is used to remove high-frequency noise from the signal.
- **Optimal Sine Wave Comparison**: The filtered signal is compared to an optimal sine wave generated from the dominant frequency of the signal.
- **Signal Statistics**: The project calculates and visualizes statistics including:
  - Root Mean Square (RMS)
  - Average value
  - Standard deviation
  - Peak-to-peak amplitude

## Requirements

To run this project, you need to install the following Python libraries:

- `pyedflib`: For reading and writing EDF files.
- `numpy`: For numerical operations.
- `scipy`: For signal processing (filtering and FFT).
- `matplotlib`: For plotting the results.

You can install the required libraries by running the following:

```shell
pip install pyedflib numpy scipy matplotlib
```


## How to Use

1. **Clone the repository**:
    ```
    git clone https://github.com/deekb/EDFTest.git
    cd signal-EDFTest
    ```

2. **Prepare your EDF file**: Ensure you have an EDF file (e.g., `waveform.edf`) in the same directory as the script or provide the correct path to the `file_path` variable in the code.

3. **Run the script**:
    ```
    python main.py
    ```

    The script will:
    - Load the EDF file.
    - Apply the low-pass filter.
    - Calculate the RMS, average, and other statistics.
    - Generate and display a plot comparing the filtered signal to the optimal sine wave.

## Output

The output will be a plot displaying:
- The **filtered signal** (in blue).
- The **optimal sine wave** (in orange) with the same frequency as the dominant frequency of the filtered signal.
- Horizontal lines showing the **RMS** and **average** of the filtered signal.

### Example Plot:

![Filtered vs Optimal Sine Wave](vector_graph.pdf)

## File Structure
signal-filtering-analysis/\
│ ├── signal_analysis.py # Main script for signal processing and analysis\
├── waveform.edf # Example EDF file (replace with your own)\
└── README.md # Project documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Notes:
- Replace the `waveform.edf` with your actual EDF file.
- Customize the `cutoff_frequency` in the code to adjust the filtering process as per your data needs.

## Contributing

Feel free to fork this repository, submit issues, and create pull requests to improve the code. Contributions are welcome!
