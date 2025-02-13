import glob as glob
import os as os
import pyabf
import numpy as np
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.io as io
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy import integrate
from sklearn import linear_model
ransac = linear_model.RANSACRegressor()
import pyabf.tools.memtest
import matplotlib.pyplot as plt

# These are the helper files that need to be in the same folder structure 
import ephys_extractor as efex
import ephys_features as ft

"""
This script contains two classes to assist with the EphysAnalyzerScript.

EphysFile: object containing all metadata for a single .abf file 

EphysFileWrapper: object containing all files to be analyzed during one call 
of the script. Groups of files (APF, MT1, MT2) are automatically detected.
"""

class EphysFileWrapper:
    """
    File wrapper that grabs all .abf files from a folderpath and organizes them by cell.

    Uses dict data type to organize files by the cell ID defined with date, region, slice number, cell number.
    
    Each key (cell ID string) will have a list of EphysFiles. Ideally, each cell has a list of 3 ephys_file objects
    pertaining to MT1, MT2, and APF for a single cell
    """
    def __init__(self, folderpath):
        self.folderpath = folderpath
        self.all_files = {}
        self.all_EphysFiles = []

        os.chdir(folderpath)
        filepaths = glob.glob("*.abf")

        for filepath in filepaths:
            abf_file = pyabf.ABF(filepath)
            current_file = EphysFile(filepath, abf_file)
            self.all_EphysFiles.append(current_file) 
        
        for file in self.all_EphysFiles:
            current_ID = file.getCellID()
            current_list = self.all_files.get(current_ID, [])
            current_list.append(file)
            self.all_files[current_ID] = current_list

    def getAllFiles(self):
        return self.all_files

class EphysFile:
    
    def __init__(self, filename, abf_data):
        """
        Metadata is automatically grabbed from the filename.
        Assumes all filenames are named in the following format:
        DATE_REGION_SLICENUMBER_CELLNUMBER_APF/MT1/MT2_ID

        If the recording type is APF, the data is parsed and stored accordingly.
        """
        values = str.split(filename, '_')
        if len(values) != 6:
            raise ValueError('Filename is incorrectly named and will be challenging to match.')
        
        self.date = values[0]
        self.region = values[1]
        self.sliceNumber = values[2]
        self.cellNumber = values[3]
        self.recordingMethod = values[4]
        self.recordingID = values[5]
        self.cellID = self.date + '_' + self.region + '_' + self.sliceNumber + '_' + self.cellNumber
        self.abf_data = abf_data

        self.voltage = []
        self.current_traces = []
        self.current = []
        self.time = []
        self.current_index_0 = []

        if self.recordingMethod == 'APF':
            voltage = np.zeros((abf_data.sweepPointCount, abf_data.sweepCount))
            current_traces = np.zeros((abf_data.sweepPointCount, abf_data.sweepCount))
            current = np.zeros((abf_data.sweepCount))
            time = abf_data.sweepX
            curr_index_0 = -1
            for sweepNumber in range(abf_data.sweepCount):
                abf_data.setSweep(sweepNumber)
                voltage[:, sweepNumber] = abf_data.sweepY
                current_traces[:, sweepNumber] = abf_data.sweepC
                
                if min(abf_data.sweepC) == max(abf_data.sweepC):
                    current[sweepNumber] = 0
                    curr_index_0 = sweepNumber
                elif min(abf_data.sweepC) < 0:
                    current[sweepNumber] = min(abf_data.sweepC)
                else:
                    current[sweepNumber] = max(abf_data.sweepC)
            
            self.voltage = voltage
            self.current_traces = current_traces
            self.current = current
            self.time = time
            self.current_index_0 = curr_index_0

    def __str__(self):
        temp_str = self.cellID + ' ' + 'Recording method: ' + self.recordingMethod
        return temp_str

    def getRecordingMethod(self):
        return self.recordingMethod
    
    def getCellID(self):
        return self.cellID
    
    def getABFRawData(self):
        return self.abf_data
    
    def getZeroIndex(self):
        return self.curr_index_zero
    
    def getABFSortedData(self):
        return self.time, self.current, self.voltage, self.current_traces, self.current_index_0
    
    def get_membrane_test(self, ABF_file):
        """
        ABF_file must have voltage clamp data
        """
        memtest = pyabf.tools.memtest.Memtest(ABF_file)
        return memtest
    
    def plot_memtest(self, abf, memtest, filename):
        """
        Used to verify that membrane test is working well.
        """
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle(filename)

        ax1 = fig.add_subplot(221)
        ax1.grid(alpha=.2)
        ax1.plot(abf.sweepTimesMin, memtest.Ih.values,
                ".", color='C0', alpha=.7, mew=0)
        ax1.set_title(memtest.Ih.name)
        ax1.set_ylabel(memtest.Ih.units)

        ax2 = fig.add_subplot(222)
        ax2.grid(alpha=.2)
        ax2.plot(abf.sweepTimesMin, memtest.Rm.values,
                ".", color='C3', alpha=.7, mew=0)
        ax2.set_title(memtest.Rm.name)
        ax2.set_ylabel(memtest.Rm.units)

        ax3 = fig.add_subplot(223)
        ax3.grid(alpha=.2)
        ax3.plot(abf.sweepTimesMin, memtest.Ra.values,
                ".", color='C1', alpha=.7, mew=0)
        ax3.set_title(memtest.Ra.name)
        ax3.set_ylabel(memtest.Ra.units)

        ax4 = fig.add_subplot(224)
        ax4.grid(alpha=.2)
        ax4.plot(abf.sweepTimesMin, memtest.CmStep.values,
                ".", color='C2', alpha=.7, mew=0)
        ax4.set_title(memtest.CmStep.name)
        ax4.set_ylabel(memtest.CmStep.units)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.margins(0, .9)
            ax.set_xlabel("Experiment Time (minutes)")
            for tagTime in abf.tagTimesMin:
                ax.axvline(tagTime, color='k', ls='--')

        plt.tight_layout()
        plt.show()

    def get_access_resistance(self, memtest):
        ra_values = memtest.Ra.values
        return np.median(ra_values)
    
    def get_membrane_resistance(self, memtest):
        rm_values = memtest.Rm.values
        return np.median(rm_values)
        

    def extract_spike_features(self, time, current, voltage, current_array_all, start = .11716, end = .71716, fil = 10):
        """ Analyse the voltage traces and extract information for every spike (returned in df), and information for all the spikes
        per current stimulus magnitude.

        Parameters
        ----------
        time : numpy 1D array of the time (s)
        current : numpy 1D array of all possible current stimulation magnitudes (pA)
        voltage : numpy ND array of all voltage traces (mV) corresponding to current stimulation magnitudes
        current_array_all: numpy ND array of current stim trace
        start : start of the stimulation (s) in the voltage trace (optional, default 0.1)
        end : end of the stimulation (s) in the voltage trace (optional, default 0.7)
        fil : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
        
        Returns
        -------
        df : DataFrame with information for every detected spike (peak_v, peak_index, threshold_v, ...)
        df_related_features : DataFrame with information for every possible used current stimulation magnitude
        """
        
        df = pd.DataFrame()
        df_related_features = pd.DataFrame()

        for c, curr in enumerate(current):  #c is an index and curr is the value of the current step 
            start_index = ft.find_time_index(time, start) # Find closest index where the injection current starts
            end_index = ft.find_time_index(time, end) # Find closest index where the injection current ends
            current_array = current_array_all[:, c]


            EphysObject = efex.EphysSweepFeatureExtractor(t = time, v = voltage[:, c], i = current_array, start = start, \
                                                        end = end, filter = fil) # Goes to ephys_extractor.py to create an object
            
            #This object processes information per sweep, so therefore requires the object to be created with one time trace, one voltage sweep, one current array
            EphysObject.process_spikes()
            
            # Adding peak_height (mV) + code for maximum frequency determination (see further)
            spike_count = 0
            if EphysObject._spikes_df.size: # If this sweep has any detected spikes
                EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                    EphysObject._spikes_df['threshold_v'].values
                spike_count = EphysObject._spikes_df['threshold_i'].values.size
            df = pd.concat([df, EphysObject._spikes_df], sort = True)

            # Some easily found extra features
            df_features = EphysObject._sweep_features

            # Adding spike count
            df_features.update({'spike_count': spike_count})
            
            # Adding spike frequency adaptation (ratio of spike frequency of second half to first half)
            SFA = np.nan
            half_stim_index = ft.find_time_index(time, float(start + (end-start)/2))
            if spike_count > 5: # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here
                                # but in the end we only take the trace with the max amount of spikes
                
                if np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] < half_stim_index)!=0:
                    SFA = np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] > half_stim_index) / \
                    np.sum(df.loc[df['threshold_i'] == curr, :]['threshold_index'] < half_stim_index)
            
            df_features.update({'SFA': SFA})
            
            # Adding current (pA)
            df_features.update({'current': curr})

            # Adding membrane voltage (mV)
            df_features.update({'resting_membrane_potential': EphysObject._get_baseline_voltage()}) #average of 100 ms before start of current injection

            # Adding voltage deflection to steady state (mV)
            voltage_deflection_SS = ft.average_voltage(voltage[:, c], time, start = end - 0.5, end = end) ## steady state voltage is last 500 ms
            df_features.update({'voltage_deflection': voltage_deflection_SS})
            
            # Adding input resistance (MOhm)
            input_resistance = np.nan
            if not ('peak_i' in EphysObject._spikes_df.keys()) and not curr==0: # We only calculate input resistances 
                                                                                # from traces without APs
                input_resistance = (np.abs(voltage_deflection_SS - EphysObject._get_baseline_voltage())*1000)/np.abs(curr)
                if input_resistance == np.inf:
                    input_resistance = np.nan
            df_features.update({'input_resistance': input_resistance})

            # Adding membrane time constant (s) and voltage plateau level for hyperpolarisation paradigms
            # after stimulus onset
            tau = np.nan
            E_plat = np.nan
            sag_ratio = np.nan
            
            if curr < 0:            # We use hyperpolarising steps as required in the object function to estimate the
                                    # membrane time constant and E_plateau
                while True:
                    try:
                        tau = EphysObject.estimate_time_constant()  # Result in seconds!
                        break
                    except TypeError: # Probably a noisy bump for this trace, just keep it to be np.nan
                        break
                E_plat = ft.average_voltage(voltage[:, c], time, start = end - 0.1, end = end)
                sag, sag_ratio = EphysObject.estimate_sag()

            df_features.update({'tau': tau}) #Current traces below zero are used for membrane time constant fitting
            df_features.update({'E_plat': E_plat})
            df_features.update({'sag_ratio': sag_ratio})
            df_features.update({'sag_amplitude': sag})
            
            # For the rebound and sag time we only are interested in the lowest (-400 pA (usually)) hyperpolarisation trace
            rebound = np.nan
            sag_time = np.nan
            sag_area = np.nan
            if c==0:
                #print('in the very first current step, to measure sag ...')
                baseline_interval = 0.1 # To calculate the SS voltage
                v_baseline = EphysObject._get_baseline_voltage()

                #print(f'Baseline voltage: {v_baseline}')
                if np.flatnonzero(voltage[end_index:, c] > v_baseline).size == 0: # So perfectly zero here means
                                                                                # it did not reach it
                    rebound = 0
                    print('thinks rebound is zero')
                else:
                    index_rebound = end_index + np.flatnonzero(voltage[end_index:, c] > v_baseline)[0]
                    if time[index_rebound] < (end + 0.15): # We definitely have 150 ms left to calculate the rebound
                        rebound = ft.average_voltage(voltage[index_rebound:index_rebound + ft.find_time_index(time, 0.15), c], \
                                            time[index_rebound:index_rebound + ft.find_time_index(time, 0.15)]) - v_baseline
                    else:# Work with whatever time is left
                        if time[-1] == time[index_rebound]:
                            rebound=0
                        else:
                            rebound = ft.average_voltage(voltage[index_rebound:, c], \
                                            time[index_rebound:]) - v_baseline

                v_peak, peak_index, v_steady, steady_index = EphysObject.voltage_deflection_sag()

                #print(f"peak voltage: {v_peak}")
                #print(f"steady voltage: {v_steady}")
                if v_peak >= v_steady:
                    sag_time = 0
                    sag_area = 0
                else:
                    #First time SS is reached after stimulus onset
                    first_index = start_index + np.flatnonzero(voltage[start_index:peak_index, c] < v_steady)[0]
                    # First time SS is reached after the max voltage deflection downwards in the sag
                    if np.flatnonzero(voltage[peak_index:end_index, c] > v_steady).size == 0: 
                        second_index = end_index 
                    else:
                        second_index = peak_index + np.flatnonzero(voltage[peak_index:end_index, c] > v_steady)[0]
                    sag_time = time[second_index] - time[first_index] #
                    sag_area = -integrate.cumulative_trapezoid(voltage[first_index:second_index, c], time[first_index:second_index])[-1]


            burst_metric = np.nan
            if spike_count > 5:
                burst = EphysObject._process_bursts()
                if len(burst) != 0:
                    burst_metric = burst[0][0]
                
            df_features.update({'rebound': rebound})
            df_features.update({'sag_time': sag_time})
            df_features.update({'sag_area': sag_area})
            df_features.update({'burstiness': burst_metric})

            df_related_features = pd.concat([df_related_features, pd.DataFrame([df_features])], sort = True)
        
        return df, df_related_features

    def get_cell_features(self, df, df_related_features, time, current, voltage, curr_index_0, \
                        current_step = 20, axis = None, start = .11716, end = .71716, cell_name = None):
        """ Analyse all the features available for the cell per spike and per current stimulation magnitude. Extract typical
        features includig the resting membrane potential (Vm, mV), the input resistance (R_input, MOhm), the membrane time constant
        (tau, ms), the action potential threshold (AP threshold, mV), the action potential amplitude (AP amplitude, mV),
        the action potential width (AP width, ms), the afterhyperpolarisation (AHP, mV), the afterdepolarisation
        (ADP, mV), the adaptation index (AI, %) and the maximum firing frequency (max freq, Hz).
        
        Parameters
        ----------
        df : DataFrame with information for every detected spike
        df_related_features : DataFrame with information for every possible used current stimulation magnitude
        time : numpy 1D array of the time (s)
        current : numpy 1D array of all possible current stimulation magnitudes (pA)
        voltage : numpy ND array of all voltage traces (mV) corresponding to current stimulation magnitudes
        curr_index_0 : integer of current index where the current = 0 pA
        current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
        axis : figure axis object (optional, None by default)
        start : start of stimulation interval (s, optional)
        end : end of stimulation interval (s, optional)
        cell_name: name of cell file for plotting rheobase
        
        Returns
        ----------
        Cell_Features : DataFrame with values for all required features mentioned above
        """

        tau_array = df_related_features['tau'][:curr_index_0].dropna().values #Only up until current index = 0 (in Seconds)
        Rm_array = df_related_features['resting_membrane_potential'][:curr_index_0].dropna().values 
        Ri_array = df_related_features['input_resistance'][:curr_index_0].dropna().values #in MegaOhms
        
        
        if len(tau_array) == len(Ri_array):
            capac_array = tau_array/Ri_array
            capac = np.median(capac_array)*pow(10, 6) #Capacitance here median (s/Mohm) converted to picoF# 
        else:
            capac = df_related_features['tau'].values[0]/df_related_features['input_resistance'].values[0]*pow(10,6) #steepest hyperpolarizing trace used

        tau = np.median(tau_array)*1000#
        Rm = np.median(Rm_array) #Resting Membrane-Potential Median of all the current steps before the 0 (all hyperpolarizing)
        Ri = np.median(Ri_array) #Input Resistance-Input resistance of all the current steps before the 0

        sag_amplitude = df_related_features['sag_amplitude'].values[0] #steepest hyperpolarising trace used
        sag_ratio = df_related_features['sag_ratio'].values[0] # Steepest hyperpolarising trace used
        rebound = df_related_features['rebound'].values[0] # Steepest hyperpolarising trace used
        sag_time = df_related_features['sag_time'].values[0] # Steepest hyperpolarising trace used
        sag_area = df_related_features['sag_area'].values[0] # Steepest hyperpolarising trace used
        
        
        if not df.empty:
            # The max amount of spikes in 600 ms of the trace showing the max amount of spikes in 600 ms
            max_freq = np.max(df_related_features['spike_count'].values)
            # Take the first trace showing this many spikes if there are many
            current_max_freq = np.flatnonzero(df_related_features['spike_count'].values >= max_freq)[0]
            
            
            
            # Rebound firing
            rebound_spikes = 0
            EphysObject_rebound = efex.EphysSweepFeatureExtractor(t = time[ft.find_time_index(time, end):], \
                                                        v = voltage[ft.find_time_index(time, end):, 0], \
                                                        i = np.zeros_like(time[ft.find_time_index(time, end):]), \
                                                        start = end, end = time[-1])
            EphysObject_rebound.process_spikes()
            if EphysObject_rebound._spikes_df.size:
                rebound_spikes = EphysObject_rebound._spikes_df['threshold_i'].values.size
            

            # Check if there are APs outside the stimilation interval for the highest firing trace. When true, continue looking
            # for lower firing traces untill it shows none anymore. We don't want the neuron to have gone 'wild' (i.e. dying).
            current_max_freq_initial = current_max_freq # to see for which cells there is going to be much of a difference
            artifact_occurence = False
            EphysObject_end = efex.EphysSweepFeatureExtractor(t = time[ft.find_time_index(time, end):], \
                                                        v = voltage[ft.find_time_index(time, end):, current_max_freq], \
                                                        i = np.zeros_like(time[ft.find_time_index(time, end):]), \
                                                        start = end, end = time[-1])
            EphysObject_start = efex.EphysSweepFeatureExtractor(t = time[:ft.find_time_index(time, start)+1], \
                                                        v = voltage[:ft.find_time_index(time, start)+1, current_max_freq], \
                                                        i = np.zeros_like(time[:ft.find_time_index(time, start)]), \
                                                        start = 0, end = start)

            EphysObject_end.process_spikes()
            EphysObject_start.process_spikes()
            if EphysObject_end._spikes_df.size or EphysObject_start._spikes_df.size:
                        artifact_occurence = True
            while artifact_occurence:
                current_max_freq-=1

                EphysObject_end = efex.EphysSweepFeatureExtractor(t = time[ft.find_time_index(time, end):], \
                                                        v = voltage[ft.find_time_index(time, end):, current_max_freq], \
                                                        i = np.zeros_like(time[ft.find_time_index(time, end):]), \
                                                        start = end, end = time[-1])
                EphysObject_start = efex.EphysSweepFeatureExtractor(t = time[:ft.find_time_index(time, start)+1], \
                                                        v = voltage[:ft.find_time_index(time, start)+1, current_max_freq], \
                                                        i = np.zeros_like(time[:ft.find_time_index(time, start)]), \
                                                        start = 0, end = start)
                EphysObject_end.process_spikes()
                EphysObject_start.process_spikes()
                if not EphysObject_end._spikes_df.size and not EphysObject_start._spikes_df.size:
                    artifact_occurence = False
            
            
            # Adding wildness: the feature that for neurogliaform cells specifically can describe whether for highest firing
            # traces the cell sometimes shows APs before and/or after the current stimulation window.
            wildness = df_related_features.iloc[current_max_freq_initial, :].loc['spike_count'] - \
                        df_related_features.iloc[current_max_freq, :].loc['spike_count']
            
            # Adding spike frequency adaptation (ratio of spike frequency of second half to first half for the highest
            # frequency count trace)
            if df_related_features.iloc[current_max_freq, :].loc['spike_count'] < 5: # If less than 5 spikes we choose not
                                                                                    # to calculate the SFA ==> np.nan
                SFA = np.nan
            else:
                SFA = df_related_features.iloc[current_max_freq, :].loc['SFA']
            # Note: we are trying to make sure that if SFA is 0, that it is actually 0 in the way it is defined 
            
            
            # Adding the Fano factor, a measure of the dispersion of a probability distribution (std^2/mean of the isis)
            # Adding the coefficient of variation. Time intervals between Poisson events should follow an exponential distribution
            # for which the cv should be 1. So if the neuron fires like a Poisson process a cv = 1 should capture that.
            fano_factor = df_related_features.iloc[current_max_freq, :].loc['fano_factor']
            cv = df_related_features.iloc[current_max_freq, :].loc['cv']
            AP_fano_factor = df_related_features.iloc[current_max_freq, :].loc['AP_fano_factor']
            AP_cv = df_related_features.iloc[current_max_freq, :].loc['AP_cv']
            
            
            # Do we have non-Nan values for the burstiness feature?
            non_nan_indexes_BI = ~np.isnan(df_related_features['burstiness'].values)
            if non_nan_indexes_BI.any():
                
                # Consider only the first and first 5 after threshold reached if possible
                # np.sum will consider a True as a 1 here and a False as a 0 (so you count the True's effectively)
                
                if np.sum(non_nan_indexes_BI) >= 5:
                    BI_array = df_related_features['burstiness'].values[non_nan_indexes_BI][0:5]
                    burstiness = np.median(BI_array)
                else: # Take everything you have
                    BI_array = df_related_features['burstiness'].values[non_nan_indexes_BI]
                    burstiness = np.median(BI_array)
                if burstiness < 0:
                    burstiness = 0
            else:
                burstiness = 0
            
            
            # Do we have non-Nan values for the adaptation index (i.e. traces with more than 1 spike)? We can use this to
            # calculate AP amplitude changes too
            non_nan_indexes_AI = ~np.isnan(df_related_features['isi_adapt'].values)
            
            if non_nan_indexes_AI.any():
                
                # Consider only the first 5 after threshold reached if possible
                # np.sum will consider a True as a 1 here and a False as a 0 (so you count the True's effectively)
                
                if  np.sum(non_nan_indexes_AI) >= 5:
                    ISI_adapt_array = df_related_features['isi_adapt'].values[non_nan_indexes_AI][0:5]
                    ISI_adapt = np.median(ISI_adapt_array)
                    ISI_adapt_average_array = df_related_features['isi_adapt_average'].values[non_nan_indexes_AI][0:5]
                    ISI_adapt_average = np.median(ISI_adapt_average_array)
                    AP_amp_adapt_array = df_related_features['AP_amp_adapt'].values[non_nan_indexes_AI][0:5]
                    AP_amp_adapt = np.median(AP_amp_adapt_array)
                    AP_amp_adapt_average_array = df_related_features['AP_amp_adapt_average'].values[non_nan_indexes_AI][0:5]
                    AP_amp_adapt_average = np.median(AP_amp_adapt_average_array)
                    
                else: # Take everything you have
                    ISI_adapt_array = df_related_features['isi_adapt'].values[non_nan_indexes_AI]
                    ISI_adapt = np.median(ISI_adapt_array)
                    ISI_adapt_average_array = df_related_features['isi_adapt_average'].values[non_nan_indexes_AI]
                    ISI_adapt_average = np.median(ISI_adapt_average_array)
                    AP_amp_adapt_array = df_related_features['AP_amp_adapt'].values[non_nan_indexes_AI]
                    AP_amp_adapt = np.median(AP_amp_adapt_array)
                    AP_amp_adapt_average_array = df_related_features['AP_amp_adapt_average'].values[non_nan_indexes_AI]
                    AP_amp_adapt_average = np.median(AP_amp_adapt_average_array)

            else:
                ISI_adapt = np.nan
                ISI_adapt_average = np.nan
                AP_amp_adapt = np.nan
                AP_amp_adapt_average = np.nan
            
            # We calculate the latency: the time it takes to elicit the first AP
            df_latency = df_related_features[df_related_features['current'] > 0]
            non_nan_indexes_latency = ~np.isnan(df_latency['latency'].values)
            # Only the first AP is considered for the first trace for which the current clamp stimulation is higher than the
            # current hold
            latency = df_latency['latency'].values[non_nan_indexes_latency][0]*1000
            latency_2 = df_latency['latency'].values[non_nan_indexes_latency][1]*1000

            # First index where there is an AP and the current stimulation magnitude is positive
            index_df =  np.where(df.loc[0]['fast_trough_i'].values[~np.isnan(df.loc[0]['fast_trough_i'].values)] > 0)[0][0]
            
            non_nan_indexes_ahp = ~np.isnan(df.loc[0]['fast_trough_v'])
            if non_nan_indexes_ahp.any():
                AHP = df.loc[0]['fast_trough_v'].values[index_df] - df.loc[0]['threshold_v'].values[index_df]
                # Only first AP is considered
            else:
                AHP = 0
        
            # ADP (alculated w.r.t. AHP)
            non_nan_indexes_adp = ~np.isnan(df.loc[0]['adp_v'])
            if non_nan_indexes_adp.any():
                ADP = df.loc[0]['adp_v'].values[index_df] - df.loc[0]['fast_trough_v'].values[index_df]
                # Only first AP is considered
            else:
                ADP = 0

            
            non_nan_indexes_thresh = ~np.isnan(df.loc[0]['threshold_v'])
            if non_nan_indexes_thresh.any():
                if df.loc[0]['threshold_v'].size > 1:
                    AP_threshold = df.loc[0]['threshold_v'].values[index_df] # TRHESHOLD CALCULATION Only first AP is considered
                    AP_amplitude = df.loc[0]['peak_height'].values[index_df] # Only first AP is considered
                    AP_width = 1000*df.loc[0]['width'].values[index_df] # Only first AP is considered
                    UDR = df.loc[0]['upstroke_downstroke_ratio'].values[index_df] # Only first AP is considered 
                else:
                    AP_threshold = df.loc[0]['threshold_v'] # Only first AP is considered
                    AP_amplitude = df.loc[0]['peak_height'] # Only first AP is considered
                    AP_width = 1000*df.loc[0]['width'] # Only first AP is considered
                    UDR = df.loc[0]['upstroke_downstroke_ratio'] # Only first AP is considered
            else:
                AP_threshold = 0
                AP_amplitude = 0
                AP_width = 0
            
            # We estimate the rheobase based on a few (i.e. 5)
            # suprathreshold currents steps. A linear fit of the spike frequency w.r.t. to the current injection values of
            # these steps should give the rheobase as the crossing with the x-axis. (Method almost in agreement with Alexandra
            # Naka et al.: "Complementary networks of cortical somatostatin interneurons enforce layer specific control.", 
            # they additionally use the subthreshold current step closest to the first suprathreshold one. We think that biases
            # the regression analysis somewhat). We take the min of the first current step showing spikes and the regression line crossing
            # with the x-axis, unless the crossing is at an intercept lower than the last current step still showing no spikes (in
            # that case the first current step showing spikes is simply taken as value for the rheoabse). This method should
            # approximate the rheobase well provided the stimulus interval is long (in our case 600 ms).
            # If a regression cannot be performed, then also simply take the first current step for which spikes have been observed 
            # (i.e. not the subthreshold current step!)
            
            # Only positive currents for this experimental paradigm (i.e. 'spikes' observed for negative currents should not
            # be there)
            df_rheobase = df_related_features[['current', 'spike_count']][df_related_features['spike_count'] > 0]
            if len(np.nonzero(df_rheobase['spike_count'].values)[0]) > 4: # More than 4 data points with spikes, meaning possible to find rheobase fit
                indices = np.nonzero(df_rheobase['spike_count'].values)[0]
                #unique_frequency_values = np.unique(df_rheobase['spike_count'].values[indices])
                #indices = [int(np.where(df_rheobase['spike_count'].values[indices] == val)[0][0]) for val in unique_frequency_values]
                if len(indices) > 4:
                    indices = indices[:5]
                    ransac.fit(df_rheobase['current'].values[indices].reshape(-1, 1), \
                            df_rheobase['spike_count'].values[indices].reshape(-1, 1)/(end-start))
                    line_X = np.concatenate((df_rheobase['current'].values[indices], \
                                            np.array([0]))).reshape(-1, 1)
                    slope = ransac.estimator_.coef_[0][0]
                    # sub_thresh_curr = df_rheobase['current'].values\
                    #     [np.nonzero(df_rheobase['spike_count'].values)[0][0] - 1] # Last current step with no observed spikes
                    first_supra_thresh_curr = df_rheobase['current'].values\
                        [np.nonzero(df_rheobase['spike_count'].values)[0][0]]  # First current step with observed spikes
                    
                    #print(sub_thresh_curr)
                    rheobase = -ransac.predict(np.array([1]).reshape(-1, 1))[0][0]/slope
                    #print(rheobase)
                    if rheobase > first_supra_thresh_curr:
                        rheobase = first_supra_thresh_curr # Take the first current step for which spikes are observed
                    
                    plot_rheobase = False
                    if plot_rheobase:
                        if axis:
                            ax = axis
                        else: 
                            figure_object, ax = plt.subplots(figsize = (10, 5))
                        ax.plot(df_rheobase['current'].values[indices].reshape(-1, 1), \
                                df_rheobase['spike_count'].values[indices].reshape(-1, 1)/(end-start), '.k', markersize = 15)
                        ax.plot(line_X, ransac.predict(line_X), color = 'k')
                        ax.set_xlabel('Current (pA)', fontsize = 17)
                        ax.set_ylabel('Spike frequency (Hz)', fontsize = 17)
                        ax.set_title(f'Rheobase estimation for cell {cell_name}', fontsize = 20)
                        ax.spines['top'].set_visible(False)
                        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
                        ax.set_ylim([0, np.max(df_rheobase['spike_count'].values[indices].reshape(-1, 1)/(end-start)) + 3])

                        # Let's denote the membrane voltage clearly on the plot
                        ax.plot(rheobase, 0, marker = "|", color = 'black', ms = 50)
                        ax.annotate('', xy = (rheobase - 15, 2), \
                                    xycoords = 'data', xytext = (rheobase, 2), \
                                    textcoords = 'data', arrowprops = {'arrowstyle': '<-', 'connectionstyle': 'arc3', \
                                                                    'lw': 2, 'ec': 'grey', 'shrinkB': 0})
                        ax.annotate('rheobase', xy = (rheobase -15, 2), \
                                    xycoords = 'data', xytext = (-15, 2), textcoords = 'offset points', color = 'k', fontsize = 15)
                    
                else: # A subset can probably not be found by RANSAC to do a regression
                    # Just take the first current step for which spikes have been observed
                    rheobase = df_rheobase['current'].values\
                                    [np.nonzero(df_rheobase['spike_count'].values)[0][0]]

            else: # Not enough datapoints to calculate a rheobase
                # Just take the first current step for which spikes have been observed
                rheobase = df_rheobase['current'].values\
                                [np.nonzero(df_rheobase['spike_count'].values)[0][0]]
            
        else:
            AHP = 0
            ADP = 0
            max_freq = 0
            rebound_spikes = 0
            SFA = np.nan
            fano_factor = np.nan
            cv = np.nan
            norm_sq_isis = np.nan
            ISI_adapt = np.nan
            ISI_adapt_average = np.nan
            AP_amp_adapt = np.nan
            AP_amp_adapt_average = np.nan
            AP_fano_factor = np.nan
            AP_cv = np.nan
            AP_threshold = 0
            AP_amplitude = 0
            AP_width = 0
            UDR = 0
            rheobase = 0
            latency = 0
            latency_2 = 0
            burstiness = 0
            
        if np.isnan(ADP):
            ADP = 0
        
        if rebound < 0:
            rebound = 0

        #Todo: Add in Capacitance value
        name_features = ['Resting membrane potential (mV)', 'Input resistance (MOhm)', 'Membrane time constant (ms)', \
                        'Cell Capacitance', 'AP threshold (mV)', 'AP amplitude (mV)', 'AP width (ms)', \
                        'Upstroke-to-downstroke ratio', 'Afterhyperpolarization (mV)', 'Afterdepolarization (mV)',
                        'ISI adaptation index', 'Max number of APs', 'Rheobase (pA)', 'Sag Amplitude' 'Sag ratio', \
                        'Latency (ms)', 'Latency @ +20pA current (ms)', 'Spike frequency adaptation', \
                        'ISI Fano factor', 'ISI coefficient of variation', \
                        'ISI average adaptation index', 'Rebound (mV)', 'Sag time (s)', 'Sag area (mV*s)', \
                        'AP amplitude adaptation index', 'AP amplitude average adaptation index', \
                        'AP Fano factor', 'AP coefficient of variation', 'Burstiness', 'Wildness', 'Rebound number of APs']
        features = [Rm, Ri, tau, capac, AP_threshold, AP_amplitude, AP_width, UDR, AHP, ADP, ISI_adapt, max_freq, rheobase, \
                    sag_amplitude, sag_ratio, latency, latency_2, SFA, fano_factor, cv, ISI_adapt_average, rebound, sag_time, \
                    sag_area, AP_amp_adapt, AP_amp_adapt_average, AP_fano_factor, AP_cv, burstiness, \
                    wildness, rebound_spikes]
        cell_features = dict(zip(name_features, features))
        Cell_Features = pd.DataFrame([cell_features])
        Cell_Features = Cell_Features.reindex(columns = name_features)
        return features, Cell_Features

#Unused code below

    # def get_baseline_current(self):
    #     if self.recordingMethod == 'APF':
    #         return -1 
        
    #     time = self.abf_data.sweepX
    #     index_end, = np.where(np.isclose(time, .004))[0]

    #     if self.abf_data.sweepCount == 1:
    #         self.abf_data.setSweep(0)
    #         current_sweep = self.abf_data.sweepC
    #         return np.average(self.abf_data.sweepC[:index_end])
    #     else:
    #         self.abf_data.setSweep(0)
    #         current_traces = np.zeros((self.abf_data.sweepPointCount, self.abf_data.sweepCount))
    #         for sweepNumber in range(self.abf_data.sweepCount):
    #             self.abf_data.setSweep(sweepNumber)
    #             current_traces[:, sweepNumber] = self.abf_data.sweepY
            
    #         average_current_trace = np.mean(current_traces, axis = 1)
    #         baseline_c = np.average(average_current_trace[:index_end])
    #         print(f'Average current: {baseline_c}')
    #         return baseline_c
        
    # def get_steady_state_current(self):
    #     if self.recordingMethod == 'APF':
    #         return -1 
        
    #     time = self.abf_data.sweepX
    #     index_start, = np.where(np.isclose(time, .02))[0]
    #     index_end, = np.where(np.isclose(time, .04))[0]

    #     if self.abf_data.sweepCount == 1:
    #         self.abf_data.setSweep(0)
    #         current_sweep = self.abf_data.sweepC
    #         return np.average(current_sweep[index_start:index_end])
    #     else:
    #         self.abf_data.setSweep(0)
    #         current_traces = np.zeros((self.abf_data.sweepPointCount, self.abf_data.sweepCount))
    #         for sweepNumber in range(self.abf_data.sweepCount):
    #             self.abf_data.setSweep(sweepNumber)
    #             current_traces[:, sweepNumber] = self.abf_data.sweepY
            
    #         average_current = np.mean(current_traces, axis = 1)
    #         plt.plot(time, average_current)
    #         plt.title('STEADYSTAE')
    #         plt.show()
    #         steady_state_c = np.average(average_current[index_start:index_end])
    #         print(f'Steady State C: {steady_state_c}')
    #         return steady_state_c
        
    # def get_peak_current(self):
    #     """
    #     Returns the positive absolute peak current for a membrane test

    #     Uses the average of all sweeps if more than one sweep
    #     """
    #     if self.recordingMethod == 'APF':
    #         return -1 
    #     time = self.abf_data.sweepX

    #     if self.abf_data.sweepCount == 1:
    #         self.abf_data.setSweep(0)
    #         return np.max(self.abf_data.sweepC)
    #     else:
    #         self.abf_data.setSweep(0)
    #         current_traces = np.zeros((self.abf_data.sweepPointCount, self.abf_data.sweepCount))
    #         for sweepNumber in range(self.abf_data.sweepCount):
    #             self.abf_data.setSweep(sweepNumber)
    #             current_traces[:, sweepNumber] = self.abf_data.sweepY
    #         average_current = np.mean(current_traces, axis = 1)
    #         plt.plot(time, average_current)
    #         #plt.show()
    #         max_values = np.max(current_traces)
    #         baseline_current = self.get_baseline_current()
    #         peak_current = np.median(max_values) - baseline_current
    #         print(peak_current)
    #         return peak_current
    
    # def get_idss(self):
    #     peak_current = self.get_peak_current()
    #     steady_state_current = self.get_steady_state_current()
    #     baseline_current = self.get_baseline_current()
    #     idss = peak_current - baseline_current
    #     print(f'IDSS: {idss}')
    #     return idss

    # def get_voltage_MT(self):

    #     self.abf_data.setSweep(0, channel=0)

    #     if self.abf_data.sweepCount == 1:
    #         self.abf_data.setSweep(0)
    #         return np.max(self.abf_data.sweepC)
    #     else:
    #         self.abf_data.setSweep(0)
    #         voltage = np.zeros((self.abf_data.sweepPointCount, self.abf_data.sweepCount))
    #         for sweepNumber in range(self.abf_data.sweepCount):
    #             voltage[:, sweepNumber] = self.abf_data.sweepC
    #         average_voltage = np.mean(voltage, axis = 1)
    #         plt.plot(average_voltage)

    #         plt.title('Voltage')
    #         plt.show()
    #         voltage_deflection = np.median(average_voltage)
    #         print(f'Voltage Deflection: {voltage_deflection}')
    #         return voltage_deflection

    # def get_access_resistance(self):
    #     max_current = self.get_peak_current()
    #     voltage_MT = self.get_voltage_MT()
    #     access_resistance = voltage_MT/max_current
    #     print(f'Access_resistance: {access_resistance}')
    #     return access_resistance
    
    # def get_membrane_resistance(self):
    #     voltage_deflection = self.get_voltage_MT()
    #     access_resistance = self.get_access_resistance()
    #     current_deflection_diff = self.get_idss()

    #     membrane_resistance = (voltage_deflection - access_resistance * current_deflection_diff)/current_deflection_diff
    #     print(f'Membrane resistance: {membrane_resistance}')
    #     return membrane_resistance
    