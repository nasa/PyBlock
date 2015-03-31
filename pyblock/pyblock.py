"""
Polarimetric Radar Beam Blockage Calculation
PyBlock v1.0


Author
------
Timothy J. Lang
NASA MSFC
timothy.j.lang@nasa.gov
(256) 961-7861


Description
-----------
    Calculates beam blockage from polarimetric radar data using the specific
differential phase (KDP) and fully self-consistent (FSC) methods of 
Lang et al. (2009). The core class is BeamBlockSingleVolume, which obtains 
blockage-relevant data from individual radar volumes. This class is invoked
repeatedly by the BeamBlockMultiVolume class to compile data from a large number
of files. Helper classes like BlockStats and RawDataStorage enable the saving,
loading, and plotting of blockage information relevant to both methods.
    If you have a set of radar files to analyze, the easiest way to start is to
import this module and start ingesting data via the BeamBlockMultiVolume class.
e.g., multiblock = pyblock.BeamBlockMultiVolume(list_of_files, **kwargs). Set 
keywords to fit the characteristics of your dataset. See DEFAULT_KW for what the
program expects and how you might change those parameters.


Last Updated
------------
v1.0 - 03/23/2015


References
----------
Giangrande, S. E., and A. V. Ryzhkov, 2005: Calibration of Dual-Polarization
    Radar in the Presence of Partial Beam Blockage. J. Atmos. Oceanic Technol., 
    22, 1156–1166. doi: http://dx.doi.org/10.1175/JTECH1766.1
Lang, T. J., S. W. Nesbitt, and L. D. Carey, 2009: On the correction of
    partial beam blockage in polarimetric radar data. J. Atmos. Oceanic Technol.,
    26, 943–957.


Dependencies
------------
numpy, pyart, csu_radartools, dualpol, warnings, os, __future__, matplotlib, gzip


Change Log
----------
v1.0 Functionality (03/23/2015)
1. Will compute beam blockage for any arbitrary set of polarimetric radar volumes
   via the KDP method of Lang et al. (2009). These results can be saved to file,
   turned into statistics, and plotted.
2. Compares KDP method's Zh and Zdr data inside and outside of blocked regions,
   in order to help with the derivation of corrections that need to be applied.
3. Collates data relevant for calculating blockage via the FSC method.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
import os
import gzip
import pickle
import pyart
import dualpol
from csu_radartools import csu_misc
from singledop import fn_timer #Remove before releasing

VERSION = '0.1'
DATA_DIR = os.sep.join([os.path.dirname(__file__), 'data'])+'/'
DEFAULT_SND = DATA_DIR + 'default_sounding.txt'
RANGE_MULT = 1000.0 #m per km
DEFAULT_RANGE = [20, 90] #km
DEFAULT_RAIN = [1.5, 2] #KDP in deg/km
DEFAULT_DRIZZ = [-0.1, 0.1] #KDP in deg/km
BAD = -32768
DEFAULT_SDP = 12
DEFAULT_BINS = 1.0
DEFAULT_ZH_UNCERTAINTY = 1.0
XTICKS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
STATS_KEYS = ['N', 'low', 'median', 'high']

#####################################

DEFAULT_KW = {'sweep': 0, 'dz': 'ZH', 'dr': 'DR', 'dp': 'DP', 'rh': 'RH',
          'kd': None, 'ld': None, 'sounding': DEFAULT_SND,
          'verbose': False, 'thresh_sdp': DEFAULT_SDP, 'fhc_T_factor': 1,
          'fhc_weights': dualpol.DEFAULT_WEIGHTS, 'fhc_name': 'FH', 'band': 'S',
          'fhc_method': 'hybrid', 'kdp_method': 'CSU', 'bad': BAD,
          'use_temp': True, 'rain_dp_thresh': 100, 'drizzle_dz_thresh': 25,
          'vr': 'VR', 'magnetron': False, 'bin_width': DEFAULT_BINS,
          'rain_kdp_thresh': DEFAULT_RAIN, 'drizzle_kdp_thresh': DEFAULT_DRIZZ,
          'rng_thresh': DEFAULT_RANGE, 'dsd_flag': False, 'output': 100,
          'rain_dz_thresh': 39, 'liquid_ice_flag': False, 'precip_flag': False}

kwargs = np.copy(DEFAULT_KW)

"""
kwargs descriptions
-------------------
sweep = Sweep number to examine
dz = Name of reflectivity field in Py-ART radar object
dr = Name of differential reflectivity field in Py-ART radar object
dp = Name of differential phase field in Py-ART radar object
rh = Name of correlation coefficient field in Py-ART radar object
kd = Name of specific differential phase field in Py-ART radar object (if avail.)
ld = Name of linear depolarization ratio field in Py-ART radar object (if avail.)
vr = Name of Doppler velocity field in Py-ART radar object
verbose = Set to True for text notifications
thresh_sdp = Threshold for specific differential phase (can vary spatially)
fhc_T_factor = Extra weighting to be used for T in FHC calculations
fhc_weights = Weights used for each polarimetric and T field in FHC calculations
fhc_name = Name to give to newly created FHC field
band = Wavelength band of radar ('S' or 'C' supported)
fhc_method = Method to use in FHC calculations
kdp_method = Method to use in KDP calculations
bad = Bad data value
use_temp = Set to False to not use T in FHC
rain_dp_thresh = Differential phase threshold below which data will not be used
                 in KDP-method blockage calculation unless reflectivity exceeds
                 rain_dz_thresh
drizzle_dz_thresh = Reflectivity threshold above which data will not be used in
                    determination of ZDR blockage magnitude
magnetron = Set to True if transmitter was a magnetron and you can thus remove
            second-trip by filtering on Doppler velocity
bin_width = Width of each bin in azimuth degrees for determining blockage
rain_kdp_thresh = Two-element tuple denoting min/max KDP values to consider in
                  estimation of reflectivity blockage via KDP method
drizzle_kdp_thresh = Two-element tuple denoting min/max KDP values to consider in
                     estimation of ZDR blockage
rng_thresh = Two-element tuple or list of tuples (same size as number of bins in
             blockage calculation) indicating range (km) to consider for analysis
             of blockage
dsd_flag = Set to True to also retrieve DSD parameters via DualPol
output = Number of files to process at a time before outputting raw data for 
         analysis
rain_dz_thresh = Reflectivity threshold below which data will not be used
                 in KDP-method blockage calculation unless differential phase
                 exceeds rain_dp_thresh
liquid_ice_flag = Set to True to also retrieve liquid/ice mass via DualPol
precip_flag = Set to True to also retrieve rainfall rate via DualPol

"""

#####################################

class BeamBlockSingleVolume(object):

    """
    Core class that processes a single volume of radar data and isolates all the 
    individual gates that meet the user-specified criteria for rain (used for 
    determining reflectivity blockage via the KDP method) as well as for drizzle
    (used for determining differential reflectivity blockage for both the KDP
    and FSC methods). An additional mask is applied to obtain good rain/drizzle
    data for the FSC method.
    
    Many kwargs are passed to dualpol.DualPolRetrieval, which is used to 
    calculate KDP (if necessary) and also do hydrometeor identification.
    
    Py-ART is used to ingest the individual radar files. The csu_radartools
    module is used to help filter bad data like insects.
    """

    def __init__(self, filename, **kwargs):
        """
        Must specify names of key polarimetric radar fields.
        KDP is optional - will calculate if needed.
        Will use default sounding provided with package if none provided.
        Expects UWYO format for soundings.
        """
        kwargs = dualpol.check_kwargs(kwargs, DEFAULT_KW)
        try:
            radar = pyart.io.read(filename)
            self.sweep = kwargs['sweep']
            self.rain_total_pts = 0
            self.drizz_total_pts = 0
            self.radar = radar.extract_sweeps([self.sweep])
            self.retrieve = dualpol.DualPolRetrieval(self.radar, **kwargs)
            self.retrieve.name_vr = kwargs['vr']
            self.thresh_sdp = kwargs['thresh_sdp']
            self.get_bad_data_mask(magnetron=kwargs['magnetron'])
            self.get_2d_azimuth_and_range()
            self.get_bins(kwargs['bin_width'])
            self.get_range_mask(kwargs['rng_thresh'])
            #For KDP Method
            self.partition_rain_data(kwargs['rain_kdp_thresh'],
                      kwargs['rain_dz_thresh'], kwargs['rain_dp_thresh'])
            self.group_rain_data()
            self.partition_drizzle_data(kwargs['drizzle_kdp_thresh'],
                                        kwargs['drizzle_dz_thresh'])
            self.group_drizzle_data()
            #For FSC Method
            self.partition_fsc_data()
            self.group_fsc_data()
        except:
            warn('Failure in reading or analyzing file, moving on ...')
            self.Fail = True
    
    def get_2d_azimuth_and_range(self):
        """
        Two-dimensionalize 1-D azimuth and range to simplify masking.
        """
        az = self.radar.azimuth['data']
        rng = self.radar.range['data'] / RANGE_MULT
        self.range, self.azimuth = np.meshgrid(rng, az)

    def get_bins(self, bin_width):
        """
        Provided an azimuth bin width, develop the azimuth bin structure.
        Assumes azimuth stays between 0 and 360 deg.
        """
        self.bin_width = bin_width
        self.nbins = get_index(360.0, bin_width)
        self.azimuth_indices = get_index(self.azimuth, bin_width)
        cond = self.azimuth_indices == self.nbins
        self.azimuth_indices[cond] = 0
    
    def get_range_mask(self, rng_thresh):
        """
        For each azimuth bin, develop a mask that will allow us to filter out
        data from undesired ranges.
        """
        test = rng_thresh[0]
        if not hasattr(test, '__len__'):
            self.fix_rng_thresh(rng_thresh)
        else:
            if len(test) != self.nbins:
                self.fix_rng_thresh(rng_thresh)
            else:
                self.range_thresh = rng_thresh
        cond = 0 * self.range
        for i in xrange(np.shape(self.range)[0]):
            subrange = self.range_thresh[self.azimuth_indices[i][0]]
            cond[i] = np.logical_or(self.range[i] < subrange[0],
                                    self.range[i] > subrange[1])
        self.range_mask = cond.astype(bool)

    def fix_rng_thresh(self, rng_thresh):
        """
        If user did not supply range thresholds that varied by azimuth, then
        populate all azimuth bins with the constant thresholds before developing
        the range mask.
        """
        dummy = []
        for i in xrange(self.nbins):
            dummy.append(rng_thresh)
        self.range_thresh = dummy

    def partition_rain_data(self, rain_kdp_thresh, rain_dz_thresh,
                            rain_dp_thresh):
        """Produces mask for all useful rain data in volume"""
        self.fhc = self.retrieve.extract_unmasked_data(self.retrieve.name_fhc)
        self.dp = self.retrieve.extract_unmasked_data(self.retrieve.name_dp)
        self.kd = self.retrieve.extract_unmasked_data(self.retrieve.name_kd)
        self.rain_not_fhc = self.fhc != 2
        self.rain_not_kdp = np.logical_or(self.kd < rain_kdp_thresh[0],
                                          self.kd > rain_kdp_thresh[1])
        self.rain_not_dzdp = np.logical_and(self.dz < rain_dz_thresh,
                                            self.dp < rain_dp_thresh)
        self.rain_good_mask = consolidate_masks([self.bad_data_mask,
             self.range_mask, self.rain_not_fhc, self.rain_not_kdp,
             self.rain_not_dzdp])
    
    def partition_drizzle_data(self, drizzle_kdp_thresh, drizzle_dz_thresh):
        """Produces mask for all useful drizzle data in volume"""
        self.drizzle_not_fhc = self.fhc != 1
        self.drizzle_not_kdp = np.logical_or(self.kd < drizzle_kdp_thresh[0],
                                             self.kd > drizzle_kdp_thresh[1])
        self.drizzle_not_dz = self.dz > drizzle_dz_thresh
        self.drizzle_good_mask = consolidate_masks([self.bad_data_mask,
             self.range_mask, self.drizzle_not_fhc, self.drizzle_not_kdp,
             self.drizzle_not_dz])
    
    def partition_fsc_data(self):
        """Consilidate and invert the bad masks, then add more good masks."""
        self.fsc_not_fhc = np.logical_and(self.fhc != 1, self.fhc != 2)
        self.fsc_good_mask = consolidate_masks([self.bad_data_mask,
                             self.range_mask, self.fsc_not_fhc])
        self.kd_mask = self.kd > 0 #Won't examine neg. Kdp regions
        self.dr_mask = self.dr > -100
        self.dz_mask = self.dz > -100
        cond = np.logical_and(self.kd_mask, self.dr_mask)
        cond = np.logical_and(self.dz_mask, cond)
        self.fsc_good_mask = np.logical_and(self.fsc_good_mask, cond)
    
    def group_rain_data(self):
        """Applies rain mask to volume, and bins up by azimuth"""
        indices = self.azimuth_indices[self.rain_good_mask]
        dzvals = self.dz[self.rain_good_mask]
        self.rain_total_pts += len(dzvals)
        self.rain_binned_dz = []
        for i in xrange(self.nbins):
           self.rain_binned_dz.append(dzvals[indices == i])
    
    def group_drizzle_data(self):
        """Applies drizzle mask to volume, and bins up by azimuth"""
        indices = self.azimuth_indices[self.drizzle_good_mask]
        drvals = self.dr[self.drizzle_good_mask]
        self.drizz_total_pts += len(drvals)
        self.drizzle_binned_dr = []
        for i in xrange(self.nbins):
           self.drizzle_binned_dr.append(drvals[indices == i])

    def group_fsc_data(self):
        """Applies FSC mask to volume, and bins up by azimuth"""
        indices = self.azimuth_indices[self.fsc_good_mask]
        dzvals = self.dz[self.fsc_good_mask]
        drvals = self.dr[self.fsc_good_mask]
        kdvals = self.kd[self.fsc_good_mask]
        self.fsc_binned_data = {}
        self.fsc_binned_data['DZ'] = []
        self.fsc_binned_data['DR'] = []
        self.fsc_binned_data['KD'] = []
        for i in xrange(self.nbins):
            self.fsc_binned_data['DZ'].append(dzvals[indices == i])
            self.fsc_binned_data['DR'].append(drvals[indices == i])
            self.fsc_binned_data['KD'].append(kdvals[indices == i])

    def get_bad_data_mask(self, magnetron=False):
        """Develops mask to identify insects, second trip, and noise"""
        self.dz = self.retrieve.extract_unmasked_data(self.retrieve.name_dz)
        self.dr = self.retrieve.extract_unmasked_data(self.retrieve.name_dr)
        self.insect_mask = csu_misc.insect_filter(self.dz, self.dr)
        if magnetron:
            vr_array = self.retrieve.extract_unmasked_data(self.retrieve.name_vr)
            self.trip2_mask = csu_misc.second_trip_filter_magnetron(vr_array)
            new_mask = np.logical_or(self.insect_mask, self.trip2_mask)
        else:
            new_mask = self.insect_mask
        if not hasattr(self.thresh_sdp, '__len__'):
            self.thresh_sdp = 0.0 * self.dz + self.thresh_sdp
        sdp_array = self.retrieve.extract_unmasked_data(self.retrieve.name_sdp)
        self.sdp_mask = csu_misc.differential_phase_filter(sdp_array,
                                                           self.thresh_sdp)
        new_mask = np.logical_or(new_mask, self.sdp_mask)
        self.bad_data_mask = new_mask

#####################################

class BlockStats(object):

    """
    Helper class that enables ingest and plotting of blockage statistics.
    """

    def __init__(self):
        """
        Purpose of this class is to provide useful common methods to other 
        classes. Hence, there is no point to populate __init__().
        """
        pass

    def load_stats(self, filename):
        """
        Load KDP-method azimuth-based blockage stats from file.
        """
        dtype={'names': ('Azimuth', 'N_r', 'low_r', 'median_r', 'high_r',
                         'N_d', 'low_d', 'median_d', 'high_d'),
             'formats': ('float', 'float', 'float', 'float', 'float',
                         'float', 'float', 'float', 'float')}
        tmp_data = np.loadtxt(filename, dtype=dtype, skiprows=2, delimiter=',')
        self.Azimuth = tmp_data['Azimuth']
        self.rain_stats = {}
        self.rain_stats['N'] = tmp_data['N_r']
        self.rain_stats['low'] = tmp_data['low_r']
        self.rain_stats['median'] = tmp_data['median_r']
        self.rain_stats['high'] = tmp_data['high_r']
        self.drizzle_stats = {}
        self.drizzle_stats['N'] = tmp_data['N_d']
        self.drizzle_stats['low'] = tmp_data['low_d']
        self.drizzle_stats['median'] = tmp_data['median_d']
        self.drizzle_stats['high'] = tmp_data['high_d']

    def make_plots(self):
        """
        Makes a two-panel plot of reflectivity and differential reflectivity
        blockage as functions of azimuth.
        """
        #Make Rainfall Reflectivity Plots
        fig = plt.figure(figsize=(7,9.5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        fig, ax1 = self._plot_medians(fig, ax1, 'rain_stats', [0,360], [0,60],
                                      '(a) KDP Method - Rainfall',
                                      'Azimuth (deg)', 'Reflectivity (dBZ)')
        fig, ax2 = self._plot_medians(fig, ax2, 'drizzle_stats', [0,360], [-4,4],
                            '(b) KDP Method - Drizzle',
                            'Azimuth (deg)', 'Differential Reflectivity (dB)')
        plt.tight_layout()
        plt.savefig(self.image_dir+'block_kdp_method'+self.image_ext)
        plt.close()

    def _plot_medians(self, fig, ax, var, xlim, ylim, title, xlabel, ylabel):
        """
        Internal method that actually produces an individual panel in the
        two-panel plot.
        """
        var = getattr(self, var)
        for j in xrange(len(self.Azimuth)):
            ax.plot([self.Azimuth[j], self.Azimuth[j]],
                    [var['low'][j], var['high'][j]], 'k-')
            ax.plot([self.Azimuth[j]], [var['median'][j]], 'bD', ms=3, mew=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(XTICKS)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

#####################################

class BeamBlockMultiVolume(BlockStats):

    """
    This class facilitates the ingest of multiple radar volumes to compute beam
    blockage as functions of individual variables (Z, ZDR) and azimuth.
    
    Key Attributes
    --------------
    rain_refl - Reflectivity data meeting rain criteria implemented by 
                BeamBlockSingleVolume, as a function of azimuth.
    drizz_zdr - Differential reflectivity data meeting drizzle criteria 
                implemented by BeamBlockSingleVolume, as a function of azimuth.
    fsc_data - Dict containing reflectivity, differential reflectivity, and 
               specific differential phase data for use with the FSC method.
               Keys are 'DZ', 'DR', and 'KD' and data are arranged by azimuth.
    Azimuth - Array of azimuth bins.
    rain_stats - Dict of rain statistics as functions of azimuth (KDP method).
    drizzle_stats - Same but for drizzle statistics (KDP/FSC method).
    """

    def __init__(self, files, image_dir='./', image_ext='.png', save=None,
                 stats_name='blockage_stats_kdp_method.txt', **kwargs):
        """
        Initializing method. This checks and populates keywords, and figures out 
        whether a list of files to process was provided, or a statistics file was
        provided. If the former, then the files will start being processed. 
        If the latter, then the statistics will be ingested and plotted.
        """
        kwargs = dualpol.check_kwargs(kwargs, DEFAULT_KW)
        self.kwargs = kwargs
        self.check_file_list(files)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.save = save
        self.stats_name = stats_name
        #If just provided a stats file, ingest it and make a plot.
        if self.stats_flag:
            print 'Read stats file, making a plot'
            self.make_plots()
        #Otherwise, if provided a list of radar volume filenames, start processing!
        else:
            self.get_sounding_list()
            self.process_multiple_volumes()
    
    def process_multiple_volumes(self):
        """
        Main processing method. This loops through the list of radar volume 
        files, iteratively calling the beam blockage correction routines supplied
        by the BeamBlockSingleVolume class, then compiling the data from each
        file into master arrays that are periodically saved to file, both as raw 
        data and as statistics, as well as plotted up.
        """
        for i, filen in enumerate(self.file_list):
            print 'i=', i, 'of', len(self.file_list)-1
            print 'radar file:', os.path.basename(filen), '& sounding:',\
                  os.path.basename(self.sounding_list[i])
            self.kwargs['sounding'] = self.sounding_list[i]
            bb = BeamBlockSingleVolume(filen, **self.kwargs)
            if not hasattr(bb, 'Fail'):
                #Initialize dicts
                if i == 0:
                    self.initialize_data_lists(bb)
                #Keep track of total points from each file
                self.rain_total_pts += bb.rain_total_pts
                self.drizz_total_pts += bb.drizz_total_pts
                print 'Total KDP method points - rain, drizzle:',\
                      self.rain_total_pts, self.drizz_total_pts
                #Populate dicts
                for j in xrange(len(self.Azimuth)):
                    if len(bb.rain_binned_dz[j]) > 0:
                        self.rain_refl[np.str(j)] =\
                            np.append(self.rain_refl[np.str(j)],
                                      bb.rain_binned_dz[j])
                    if len(bb.drizzle_binned_dr[j]) > 0:
                        self.drizz_zdr[np.str(j)] =\
                            np.append(self.drizz_zdr[np.str(j)],
                                      bb.drizzle_binned_dr[j])
                    if len(bb.fsc_binned_data['DZ'][j]) > 0:
                        self.fsc_data['DZ'][np.str(j)] =\
                            np.append(self.fsc_data['DZ'][np.str(j)],
                                      bb.fsc_binned_data['DZ'][j])
                        self.fsc_data['DR'][np.str(j)] =\
                            np.append(self.fsc_data['DR'][np.str(j)],
                                      bb.fsc_binned_data['DR'][j])
                        self.fsc_data['KD'][np.str(j)] =\
                            np.append(self.fsc_data['KD'][np.str(j)],
                                      bb.fsc_binned_data['KD'][j])
                #Periodic saving/plotting of data & statistics
                if i % 10 == 0 or i == len(self.file_list)-1:
                    print 'Saving statistics to file and plotting an image'
                    self.get_statistics()
                    self.make_plots()
                    self.write_stats()
                if i % self.kwargs['output'] == 0 or i == len(self.file_list)-1:
                    if i != 0: #Don't bother saving data if we just started
                        print 'Writing raw data to file'
                        store = RawDataStorage(obj=self, filename=self.save)
            print

    def initialize_data_lists(self, bb):
        """
        Initialize data lists. Basic structure is a 1-D array of 1-D variable-
        length lists. The master array is the same size as the number azimuth
        analysis bins (e.g., BeamBlockSingleVolume.bin_width = 1 means size 360).
        The secondary lists, which are unique to each azimuth bin, can be of any
        length and that depends on the amount of data meeting the specified 
        criteria in each bin. This will typically grow as the number of files
        processed increases.
        
        The save attribute basically flags whether the user will start from
        scratch with a list of radar volume files, or if initial arrays will be
        populated with blockage data from a previous run (e.g., on a different
        dataset from the same radar).
        """
        try:
            store = RawDataStorage(filename=self.save)
            self.Azimuth = store.Azimuth
            self.rain_refl = store.rain_refl
            self.drizz_zdr = store.drizz_zdr
            self.fsc_data = store.fsc_data
            self.rain_total_pts = 0
            self.drizz_total_pts = 0
            for key in self.rain_refl.keys():
                self.rain_total_pts += len(self.rain_refl[key])
                self.drizz_total_pts += len(self.drizz_zdr[key])
            if bb.nbins != len(self.Azimuth):
                warn('Wrong number of azimuth bins, going to #FAIL')
        except:
            self.rain_total_pts = 0
            self.drizz_total_pts = 0
            self.rain_refl = {}
            self.drizz_zdr = {}
            self.fsc_data = {}
            self.fsc_data['DZ'] = {}
            self.fsc_data['DR'] = {}
            self.fsc_data['KD'] = {}
            self.Azimuth = np.zeros(bb.nbins)
            for j in xrange(len(self.Azimuth)):
                self.rain_refl[np.str(j)] = []
                self.drizz_zdr[np.str(j)] = []
                self.fsc_data['DZ'][np.str(j)] = []
                self.fsc_data['DR'][np.str(j)] = []
                self.fsc_data['KD'][np.str(j)] = []
                self.Azimuth[j] = bb.bin_width * j

    def get_statistics(self):
        """
        Method to compute blockage statistics as functions of azimuth. Focus
        is on median values along with 95% confidence intervals.
        """
        length = len(self.Azimuth)
        self.rain_stats = {}
        self.drizzle_stats = {}
        for key in STATS_KEYS:
            self.rain_stats[key] = np.zeros(length)
            self.drizzle_stats[key] = np.zeros(length)
        for j in xrange(length):
            self.rain_stats['N'][j], self.rain_stats['low'][j], \
               self.rain_stats['median'][j], self.rain_stats['high'][j] =\
               calc_median_ci(self.rain_refl[np.str(j)])
            self.drizzle_stats['N'][j], self.drizzle_stats['low'][j], \
               self.drizzle_stats['median'][j], self.drizzle_stats['high'][j] =\
               calc_median_ci(self.drizz_zdr[np.str(j)])

    def write_stats(self):
        """
        Method to write out a blockage statistics file.
        """
        fileobj = open(self.image_dir+self.stats_name, 'w')
        wstr = 'Azimuth,#Samples(Rain),CI_LO(Rain),Median(Rain),CI_HI(Rain),'+\
               '#Samples(Drizz),CI_LO(Drizz),Median(Drizz),CI_HI(Drizz)'
        fileobj.write(wstr+'\n')
        fileobj.write('--------------------------------------------------------')
        for j in xrange(len(self.Azimuth)):
            wstr = '\n'+str(self.Azimuth[j])+','+str(self.rain_stats['N'][j])+','+\
                   str(self.rain_stats['low'][j])+','+\
                   str(self.rain_stats['median'][j])+','+\
                   str(self.rain_stats['high'][j])+','+\
                   str(self.drizzle_stats['N'][j])+','+\
                   str(self.drizzle_stats['low'][j])+','+\
                   str(self.drizzle_stats['median'][j])+','+\
                   str(self.drizzle_stats['high'][j])
            fileobj.write(wstr)
        fileobj.close()
    
    def check_file_list(self, files):
        """
        Checks first if stats file, then if not checks if argument is list or
        a single basestring (i.e., one file)
        """
        try:
            self.load_stats(files)
            self.stats_flag = True
        except:
            self.stats_flag = False
            if isinstance(files, basestring):
                self.file_list = [files]
            else:
                self.file_list = files

    def get_sounding_list(self):
        """ 
        User is responsible for inputing properly formatted string
        file names to program. Limited error checking is done.
        """
        sounding = self.kwargs['sounding']
        if isinstance(sounding, basestring):
            sndlist = []
            for i in xrange(len(self.file_list)):
                sndlist.append(sounding)
        else:
            if len(sounding) != len(file_list):
                sndlist = []
                for i in xrange(len(file_list)):
                    sndlist.append(sounding[0])
            else:
                sndlist = sounding
        self.sounding_list = sndlist

#####################################

class RawDataStorage(object):

    """
    Class that facilitates saving and loading of azimuth-binned reflectivity
    and differential reflectivity data. Uses pickle module and stores the saved
    object as a binary file.
    """

    def __init__(self, obj=None, filename=None):
        """
        Determine what was just passed to class.
        """
        if obj is not None:
            if hasattr(obj, 'kwargs'):
                self.save_raw_data(obj, filename)
            else:
                warn('Not sure was passed a proper BeamBlockMultiVolume object')
        else:
            self.load_raw_data(filename)

    def save_raw_data(self, obj, filename):
        """
        Saves data to pickled file.
        """
        self.populate_attributes(obj)
        if filename is None:
            filename = './temporary_block_data.dat'
        with gzip.open(filename+'.gz', 'wb') as f:
            pickle.dump(self, f)

    def load_raw_data(self, filename):
        """
        Loads data from pickled file.
        """
        if filename is None:
            filename = './temporary_block_data.dat'
        with gzip.open(filename+'.gz', 'rb') as f:
            loadobj = pickle.load(f)
        self.populate_attributes(loadobj)

    def populate_attributes(self, obj):
        """
        Only certain attributes are actually saved.
        """
        self.rain_refl = obj.rain_refl
        self.drizz_zdr = obj.drizz_zdr
        self.Azimuth = obj.Azimuth
        self.fsc_data = obj.fsc_data

#####################################

class MaskHelper(object):

    """Helper class to feed common methods for masking data to other classes"""

    def __init__(self):
        pass

    def get_azimuth_mask(self, azimuths=[(0,360)]):
        """
        azimuths = list of tuples, each containing a span of unblocked azimuths
                   to consider in the determination of the median Zh to compare
                   against blocked azimuths.
        """
        for i, interval in enumerate(azimuths):
            new_cond = np.logical_and(self.Azimuth >= interval[0],
                                      self.Azimuth < interval[1])
            if i == 0:
                cond = new_cond
            else:
                cond = np.logical_or(cond, new_cond)
        self.azimuth_mask = cond

#####################################

class KdpMethodAnalysis(BlockStats, MaskHelper):

    def __init__(self, filename, azimuths=[(0,360)]):
        """azimuths = list of 2-element tuples defining blocked azimuths"""
        self.load_stats(filename)
        self.get_azimuth_mask(azimuths)
        self.get_unblocked_medians()
        self.isolate_blocked_data()
        self.calc_blockage_correction()

    def get_unblocked_medians(self):
        """
        Find median values and their confidence intervals in unblocked regions.
        """
        self.unblocked_rain_stats = {}
        self.unblocked_drizz_stats = {}
        for key in STATS_KEYS:
            self.unblocked_rain_stats[key] = \
                         self.rain_stats[key][self.azimuth_mask]
            self.unblocked_drizz_stats[key] = \
                         self.drizzle_stats[key][self.azimuth_mask]
        self.rain_N, self.rain_low, self.rain_median, self.rain_high = \
            calc_median_ci(self.unblocked_rain_stats['median'])
        self.drizz_N, self.drizz_low, self.drizz_median, self.drizz_high = \
            calc_median_ci(self.unblocked_drizz_stats['median'])

    def isolate_blocked_data(self):
        """
        Isolate data within user-defined blocked regions.
        """
        self.blocked_rain_stats = {}
        self.blocked_drizz_stats = {}
        for key in STATS_KEYS:
            self.blocked_rain_stats[key] = \
                         self.rain_stats[key][~self.azimuth_mask]
            self.blocked_drizz_stats[key] = \
                         self.drizzle_stats[key][~self.azimuth_mask]
        self.blocked_azimuths = self.Azimuth[~self.azimuth_mask]

    def calc_blockage_correction(self):
        self.length = len(self.blocked_azimuths)
        self.calc_zh_correction()
        self.calc_zdr_correction()
        
    def calc_zh_correction(self):
        """
        Calculate the suggested reflectivity corrections. There are currently 
        three approaches, which will tend to agree in well-behaved data (i.e.,
        confidence intervals are narrow). In data with wide confidence intervals,
        some corrections may not end up being suggested due to lack of certainty.
        standard: Difference between median reflectivity in blocked azimuth
                  and median of unblocked azimuths is > 1 dBZ (default) and
                  the difference is greater than half the 95% confidence interval
                  at that azimuth.
        loose: standard conditions apply plus the difference between the high 
               value in the 95% interval at the blocked azimuth and the median in
               unblocked azimuths is still greater than half the unblocked
               confidence interval.
        strict: Difference between median reflectivity in blocked azimuth                     
                and median of unblocked azimuths is > 1 dBZ (default) and the 
                difference is greater than the the difference between the high 
                value in the 95% interval at the blocked azimuth and the median 
                in unblocked azimuths.
        """
        self.suggested_zh_corrections = {}
        self.suggested_zh_corrections['azimuth'] = self.blocked_azimuths
        self.suggested_zh_corrections['standard'] = np.zeros(self.length)
        self.suggested_zh_corrections['strict'] = np.zeros(self.length)
        self.suggested_zh_corrections['loose'] = np.zeros(self.length)
        self.zh_difference = self.blocked_rain_stats['median'] - self.rain_median
        self.zh_conf_bl = self.blocked_rain_stats['high'] - \
                          self.blocked_rain_stats['low']
        self.zh_diff_ci_hi = self.blocked_rain_stats['high'] - self.rain_median
        self.zh_conf_unbl = np.max(self.unblocked_rain_stats['high'] - \
                                   self.unblocked_rain_stats['low'])
        for i, az in enumerate(self.blocked_azimuths):
            if self.zh_difference[i] <= -1.0 * DEFAULT_ZH_UNCERTAINTY:
                if np.abs(self.zh_difference[i]) > 0.5 * self.zh_conf_bl[i]:
                    self.suggested_zh_corrections['standard'][i] = \
                           -1.0 * self.zh_difference[i]
                    if np.abs(self.zh_diff_ci_hi[i]) > 0.5 * self.zh_conf_unbl:
                        self.suggested_zh_corrections['loose'][i] = \
                                -1.0 * self.zh_difference[i]
                if np.abs(self.zh_difference[i]) > np.abs(self.zh_diff_ci_hi[i]):
                   self.suggested_zh_corrections['strict'][i] = \
                           -1.0 * self.zh_difference[i]
            print az, ["%.2f" % np.round(self.suggested_zh_corrections[key][i],
                       decimals=2) for key in
                       self.suggested_zh_corrections.keys() if not
                       key == 'azimuth']
    
    def calc_zdr_correction(self):
        self.suggested_zdr_corrections = {}
        self.suggested_zdr_corrections['azimuth'] = self.blocked_azimuths
        self.zdr_difference = self.drizz_median - \
                              self.blocked_drizz_stats['median']

    #Methods to determine ZDR deviation beyond confidence interval and
    #incorporate that into blockage estimation
    #Test w/ previously determined statistics from IDL programs -
    #interface class?

#####################################

class SelfConsistentAnalysis(MaskHelper):

    """
    Class to facilitate the diagnosis of partial beam blockage via the fully
    self-consistent (FSC) method as applied in Lang et al. (2009). The anchor
    reference for this technique is Giangrande and Ryzhkov (2005).
    """

    def __init__(self, filename):
        data = RawDataStorage(filename=filename)
        self.fsc_data = data.fsc_data
        self.Azimuth = data.Azimuth

    def regress_unblocked_data(self, azimuths=[(0,360)]):
        """
        azimuths = list of tuples, each containing a span of unblocked azimuths
                   to consider in the determination of the rainfall 
                   self-consistency relationship between Z, Zdr, and Kdp.
        """
        self.get_azimuth_mask(azimuths)
        self.azimuth_indices = np.int32(np.arange(len(self.Azimuth)))
        self.unblocked_azimuths = self.Azimuth[self.azimuth_mask]
        self.unblocked_azimuth_indices = self.azimuth_indices[self.azimuth_mask]
        self.populate_unblocked_data()
        logKdp = np.log10(self.unblocked_data['KD'])
        self.beta_hat = multiple_linear_regression(self.unblocked_data['DZ'],
                        [logKdp, self.unblocked_data['DR']])

    def populate_unblocked_data(self):
        self.unblocked_data = {}
        self.unblocked_data['DZ'] = []
        self.unblocked_data['DR'] = []
        self.unblocked_data['KD'] = []
        for index in self.unblocked_azimuth_indices:
            self.unblocked_data['DZ'] = np.append(self.unblocked_data['DZ'],
                self.fsc_data['DZ'][str(index)])
            self.unblocked_data['DR'] = np.append(self.unblocked_data['DR'],
                self.fsc_data['DR'][str(index)])
            self.unblocked_data['KD'] = np.append(self.unblocked_data['KD'],
                self.fsc_data['KD'][str(index)])
        #Simple check to make sure everything is the same size
        if len(self.unblocked_data['KD']) != len(self.unblocked_data['DZ']) or\
           len(self.unblocked_data['KD']) != len(self.unblocked_data['DR']) or\
           len(self.unblocked_data['DZ']) != len(self.unblocked_data['DR']):
            wstr = str(self.unblocked_data['DZ']) + ' ' +\
                   str(self.unblocked_data['DR']) + ' ' +\
                   str(self.unblocked_data['KD'])
            warn('DZ DR KD not equal length, going to #FAIL: ' + wstr)

    #Methods to do integrations and compare them to determine blockage

#####################################

#####################################

#####################################

def consolidate_masks(bad_masks):
    """
    bad_masks = list of bad data masks to apply to array.
    Returns a mask that will provide all good data.
    """
    if not hasattr(bad_masks, '__len__'):
        return ~bad_masks
    else:
        for i, mask in enumerate(bad_masks):
            if i == 0:
                new_mask = mask
            else:
                new_mask = np.logical_or(mask, new_mask)
        return ~new_mask

def get_index(azimuth, bin_width):
    """
    Obtains azimuth bin indices, provided a bin_width. Rounds to nearest integer.
    For example, for a 1-deg width, the 30-deg azimuth bin spans 29.5-30.5 deg.
    """
    index = azimuth / bin_width
    return np.int32(np.rint(index))

def calc_median_ci(array):
    """Calculate 95% confidence interval for median"""
    Nint = len(array)
    if Nint > 1:
        N = np.float(Nint)
        median = np.median(array)
        m1 = np.floor(0.5*(N-1.0))
        new_array = sorted(array)
        interv = 1.96 * np.sqrt(N) / 2.0
        index = np.int32(np.rint(np.float(m1) - interv))
        if index < 0:
            index = 0
        lo = new_array[index]
        index = np.int32(np.rint(np.float(m1+1) + interv))
        if index >= Nint:
            index = Nint - 1
        hi = new_array[index]
        #sigma = np.std(array)
        #lo = median - 1.96 * 1.25 * sigma / np.sqrt(N)
        #hi = median + 1.96 * 1.25 * sigma / np.sqrt(N)
        return Nint, lo, median, hi
    else:
        return Nint, BAD, BAD, BAD

@fn_timer
def multiple_linear_regression(independent_var, dependent_vars):
    y = independent_var
    x = dependent_vars
    X = np.column_stack(x+[[1]*len(x[0])])
    beta_hat = np.linalg.lstsq(X,y)[0]
    return beta_hat

#####################################

#####################################





