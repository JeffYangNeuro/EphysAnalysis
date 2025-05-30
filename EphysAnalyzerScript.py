# Import pacakges into this script
import sys
import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askdirectory

# These are the helper files that need to be in the same folder structure 
import ephys_extractor as efex
import ephys_features as ft
import ABFFileClass as file_wrapper

"""
Order of all features to be extracted PER cell in order that they will appear
during and after analysis
"""

NAME_FEATURES = ['Cell name', 'Resting membrane potential (mV)', 'Input resistance (MOhm)', 'Membrane time constant (ms)', \
                     'Cell Capacitance (pF)', 'AP threshold (mV)', 'AP amplitude (mV)', 'AP width (ms)', \
                     'Upstroke-to-downstroke ratio', 'Afterhyperpolarization (mV)', 'Afterdepolarization (mV)',
                     'ISI adaptation index', 'Max number of APs', 'Rheobase (pA)', 'Sag Amplitude', 'Sag ratio', \
                     'Latency (ms)', 'Latency @ +20pA current (ms)', 'Spike frequency adaptation', \
                     'ISI Fano factor', 'ISI coefficient of variation', \
                     'ISI average adaptation index', 'Rebound (mV)', 'Sag time (s)', 'Sag area (mV*s)', \
                     'AP amplitude adaptation index', 'AP amplitude average adaptation index', \
                     'AP Fano factor', 'AP coefficient of variation', 'Burstiness', 'Wildness', 'Rebound number of APs', \
                     'Access resistance MT1', 'Access resistance MT2', 'Membrane resistance MT1', 'Membrane resistance MT2', \
                     'Access/Input Percentage MT1', 'Access/Input Percentage MT2', 'Access Resistance Percent Change']


#Test comment to push 

### METHODS ###
"""
Helper methods used by the main method
"""
def grab_names():
    """
    Asks user for folder to be analyzed. Throws an exception if 
    the folder has no ABF files inside it.
    Asks user for the name of the CSV file to be exported.
    """
    folder_path = askdirectory(title='Select Folder') # shows dialog box and return the path
    csv_name = input("Please input what you want to title your excel output (ie. CellFeatures): ")
    try:
        os.chdir(folder_path)
    except OSError:
        print(f"Could not open/read file: ", folder_path)
        print("Try the script again with a different foldername or check spacing!")
        sys.exit()

    return folder_path, csv_name

def grab_ephys_files(folder_path):
    """
    Returns object with all ABF files for all cells in the folder path. 

    Uses helper class EphysFileWrapper in the ABFFileClass script
    """
    ephys_files = file_wrapper.EphysFileWrapper(folder_path)
    return ephys_files


def analyze_all_cells(ephys_files, plot_MT):
    """
    Loops through all unique cells in ephys_files to analyze all attached ABF files.
    Expected that all unique cells have (1) APF file, (2) MT1, (3) MT2.
    Uses helper methods defined in ABFFileClass that depend on ephys_extractor and ephys_features.
    
    ephys_files: EphysFileWrapper object with all files to be analyzed
    plot_MT: 'y' or 'n' for plotting MT during analysis

    Access resistance unit is MOhm
    Membrane resistance unit is MOhm
    
    Returns pandas dataframe with all calculated values.
    """
    num_files = len(ephys_files.getAllFiles())
    print(f'Found {num_files} unique cells in the selected folder to analyze.')

    features_all = []
    all_files = ephys_files.getAllFiles()

    counter = 1
    for current_cell in all_files.keys():
        print(f'{counter}/{num_files} cells \t Currently analyzing cell {current_cell}')
        abf_files = all_files[current_cell]
        features = []
        
        access_resistance_initial = -1 #Setting these values to be negative in case there is no MT1 or MT2 File
        access_resistance_end = -1
        membrane_resistance_initial = -1
        membrane_resistance_end = -1 

        has_MT1 = False
        has_MT2 = False
        for current_abf in abf_files:

            if current_abf.getRecordingMethod() == 'APF':
                time, current, voltage, current_traces, curr_index_0 = current_abf.getABFSortedData()
                df, df_related_features = current_abf.extract_spike_features(time, current, voltage, current_traces, .11716, .71716)
                features_cell, x = current_abf.get_cell_features(df, df_related_features, time, current, voltage, curr_index_0, start = .11716, end = .71716, cell_name = current_cell)
                features_cell.insert(0, current_cell)
                features = features_cell + features
                print(f'APF data for cell {current_cell} analyzed')
                
            elif current_abf.getRecordingMethod() == 'MT1':
                has_MT1 = True
                memtest = current_abf.get_membrane_test(current_abf.getABFRawData())
                access_resistance_initial = current_abf.get_access_resistance(memtest)
                membrane_resistance_initial = current_abf.get_membrane_resistance(memtest)
                if plot_MT == 'y':
                    current_abf.plot_memtest(current_abf.getABFRawData(), memtest, current_abf)
                print(f'MT1 data for cell {current_cell} analyzed')

            elif current_abf.getRecordingMethod() == 'MT2': 
                has_MT2 = True
                memtest = current_abf.get_membrane_test(current_abf.getABFRawData())
                access_resistance_end = current_abf.get_access_resistance(memtest)
                membrane_resistance_end = current_abf.get_membrane_resistance(memtest)
                if plot_MT == 'y':
                    current_abf.plot_memtest(current_abf.getABFRawData(), memtest, current_abf)
                print(f'MT2 data for cell {current_cell} analyzed')

        counter = counter + 1
        
        features.append(access_resistance_initial)
        features.append(access_resistance_end)
        features.append(membrane_resistance_initial)
        features.append(membrane_resistance_end)

        if has_MT1:
            features.append((access_resistance_initial/membrane_resistance_initial) * 100)
        else:
            features.append(-1)
        
        if has_MT2: 
            features.append((access_resistance_end/membrane_resistance_end) * 100)
        else:
            features.append(-1)
        
        if has_MT1 and has_MT2:
            features.append((access_resistance_end - access_resistance_initial) * 100 /access_resistance_initial)
        else:
            features.append(-1)
        features_all.append(features)

    Cell_Features = pd.DataFrame(features_all, columns = NAME_FEATURES)
    return Cell_Features

def export_to_CSV(csv_name, folder_path, Cell_Features):
    """
    Exports pandas dataframe to csv file in the provided folder path with csv_name
    """
    csvname_full = csv_name + '.csv'
    full_filename = os.path.join(folder_path, csvname_full)
    print(f'Exporting all calculated values to {full_filename}')
    Cell_Features.to_csv(full_filename)


def main():
    """
    Called when the script is run to find, analyze, and export the data.
    """
    folder_path, csv_name = grab_names()
    ephys_files = grab_ephys_files(folder_path)
    
    valid_answer = False
    plot_MT = []
    while not valid_answer:
        plot_MT = input('Do you want to plot the membrane tests during analysis? (y/n): ')
        if plot_MT == 'y' or plot_MT == 'n':
            valid_answer = True
        else:
            print('Invalid input, please try again :o')
    
    Cell_Features = analyze_all_cells(ephys_files, plot_MT)
    export_to_CSV(csv_name, folder_path, Cell_Features)


if __name__ == "__main__":
    main() 