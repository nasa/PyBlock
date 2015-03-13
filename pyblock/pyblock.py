"""
Polarimetric Radar Beam Blockage Calculation
PyBlock v1.0

Author
------
Timothy Lang
timothy.j.lang@nasa.gov

Description
-----------
Calculates beam blockage from polarimetric radar data using the specific
differential phase (KDP) method of Lang et al. (2009).

Last Updated
------------
v1.0 - 03/13/2015

Reference
---------
Lang, T. J., S. W. Nesbitt, and L. D. Carey, 2009: On the correction of 
partial beam blockage in polarimetric radar data. J. Atmos. Oceanic Technol., 
26, 943–957.

Dependencies
------------
numpy, pyart, csu_radartools, dualpol, warnings, os, __future__, matplotlib

Change Log
----------
v1.0 Functionality (03/13/2015)
1. Will compute beam blockage for any arbitrary set of polarimetric radar volumes
   via the KDP method of Lang et al. (2009). These results can be saved to file,
   turned into statistics, and plotted.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pyart
from warnings import warn
from csu_radartools import csu_misc
import dualpol
#from singledop import fn_timer

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

#####################################

DEFAULT_KW = {'sweep': 0, 'dz': 'ZH', 'dr': 'DR', 'dp': 'DP', 'rh': 'RH',
          'kd': None, 'ld': None, 'sounding': DEFAULT_SND,
          'verbose': False, 'thresh_sdp': DEFAULT_SDP, 'fhc_T_factor': 1,
          'fhc_weights': dualpol.DEFAULT_WEIGHTS, 'fhc_name': 'FH', 'band': 'S',
          'fhc_method': 'hybrid', 'kdp_method': 'CSU', 'bad': BAD,
          'use_temp': True, 'rain_dp_thresh': 100, 'drizzle_dz_thresh': 25,
          'vr': 'VR', 'magnetron': False, 'bin_width': DEFAULT_BINS,
          'rain_kdp_thresh': DEFAULT_RAIN, 'drizzle_kdp_thresh': DEFAULT_DRIZZ,
          'rng_thresh': DEFAULT_RANGE, 'block_method': 'KDP',
          'rain_dz_thresh': 39, 'liquid_ice_flag': False, 'precip_flag': False,
          'dsd_flag': False}

kwargs = np.copy(DEFAULT_KW)

#####################################

class BeamBlockSingleVolume(object):

    """
    Core class that processes a single volume of radar data and isolates all the 
    individual gates that meet the user-specified criteria for rain (used for 
    determining reflectivity blockage via the KDP method) as well as for drizzle
    (used for determining differential reflectivity blockage for both the KDP
    and FSC methods).
    
    Many kwargs are passed to dualpol.DualPolRetrieval, which is used to 
    calculate KDP (if necessary) and also do hydrometeor identification.
    
    Py-ART is used to ingest the individual radar files. The csu_radartools
    module is used to help filter bad data like insects.
    
    TO DO
    -----
    1. FSC method implementation
    """

    def __init__(self, filename, **kwargs):
        """
        Must specify names of key polarimetric radar fields.
        KDP is optional - will calculate if needed.
        Will use default sounding provided with package if none provided.
        Expects UWYO format for soundings.
        """
        kwargs = dualpol.check_kwargs(kwargs, DEFAULT_KW)
        radar = pyart.io.read(filename)
        self.sweep = kwargs['sweep']
        self.radar = radar.extract_sweeps([self.sweep])
        self.retrieve = dualpol.DualPolRetrieval(self.radar, **kwargs)
        self.retrieve.name_vr = kwargs['vr']
        self.thresh_sdp = kwargs['thresh_sdp']
        self.get_bad_data_mask(magnetron=kwargs['magnetron'])
        self.get_2d_azimuth_and_range()
        self.get_bins(kwargs['bin_width'])
        self.get_range_mask(kwargs['rng_thresh'])
        if kwargs['block_method'].upper() == 'KDP':
            self.partition_rain_data(kwargs['rain_kdp_thresh'],
                  kwargs['rain_dz_thresh'], kwargs['rain_dp_thresh'])
            self.group_rain_data()
        self.partition_drizzle_data(kwargs['drizzle_kdp_thresh'],
                                    kwargs['drizzle_dz_thresh'])
        self.group_drizzle_data()
    
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

    def partition_rain_data(self, rain_kdp_thresh, rain_dz_thresh, rain_dp_thresh):
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
    
    def group_rain_data(self):
        """Applies rain mask to volume, and bins up by azimuth"""
        indices = self.azimuth_indices[self.rain_good_mask]
        dzvals = self.dz[self.rain_good_mask]
        self.rain_binned_dz = []
        for i in xrange(self.nbins):
           self.rain_binned_dz.append(dzvals[indices == i])
    
    def group_drizzle_data(self):
        """Applies drizzle mask to volume, and bins up by azimuth"""
        indices = self.azimuth_indices[self.drizzle_good_mask]
        drvals = self.dr[self.drizzle_good_mask]
        self.drizzle_binned_dr = []
        for i in xrange(self.nbins):
           self.drizzle_binned_dr.append(drvals[indices == i])
    
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
    Core class that enables ingest and plotting of blockage statistics.
    """

    def __init__(self):
        """
        Purpose of this class is to provide useful methods to other classes.
        Hence, there is no point to opulate __init__().
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
                                      '(a) KDP Method - Rainfall', 'Azimuth (deg)',
                                      'Reflectivity (dBZ)')
        fig, ax2 = self._plot_medians(fig, ax2, 'drizzle_stats', [0,360], [-4,4],
                                      '(b) KDP Method - Drizzle', 'Azimuth (deg)',
                                      'Differential Reflectivity (dB)')
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
    drizz_zdr - Differential reflectivity data meeting drizzle criteria implemented
                by BeamBlockSingleVolume, as a function of azimuth.
    Azimuth - Array of azimuth bins.
    rain_stats - Dictionary of rain statistics as functions of azimuth.
    drizzle_stats - Same but for drizzle statistics.

    TO DO
    -----
    1. FSC implementation
    """

    def __init__(self, files, image_dir='./', image_ext='.png', save=None,
                 **kwargs):
        """
        Initializing method. This checks and populates keywords, and figures out 
        whether a list of files to process was provided, or a statistics file was
        provided. If the former, then the files will start being processed. If the
        latter, then the statistics will be ingested and plotted.
        """
        kwargs = dualpol.check_kwargs(kwargs, DEFAULT_KW)
        self.kwargs = kwargs
        self.check_file_list(files)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.save = save
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
        Main processing method. This loops through the list of radar volume files,
        iteratively calling the beam blockage correction routines supplied by the
        BeamBlockSingleVolume class, then compiling the data from each file
        into master arrays that are periodically saved to file, both as raw data
        and as statistics, as well as plotted up.
        """
        for i, filen in enumerate(self.file_list):
            print 'radar file:', os.path.basename(filen), '& sounding:',\
                  os.path.basename(self.sounding_list[i])
            self.kwargs['sounding'] = self.sounding_list[i]
            bb = BeamBlockSingleVolume(filen, **self.kwargs)
            #Initialize dicts
            if i == 0:
                self.initialize(bb)
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
            #Periodic saving/plotting of data & statistics
            if i % 10 == 0 or i == len(self.file_list)-1:
                print 'i=', i, 'of', len(self.file_list)-1
                self.get_statistics()
                self.make_plots()
                self.write_stats()
                store = RawDataStorage(obj=self, filename=self.save)

    def initialize(self, bb):
        """
        Initialize data lists. Basic structure is a 1-D array of 1-D variable-
        length lists. The master array is the same size as the number azimuth
        analysis bins (e.g., BeamBlockSingleVolume.bin_width = 1.0 means size 360).
        The secondary lists, which are unique to each azimuth bin, can be of any
        length and that depends on the amount of data meeting the specified 
        criteria in each bin. This will typically grow as the number of files
        processed increases.
        
        The save attribute basically flags whether the user will start from
        scratch with a list of radar volume files, or if the initial arrays will be
        populated with blockage data from a previous run (e.g., on a different
        dataset from the same radar).
        """
        if self.save is None:
            self.rain_refl = {}
            self.drizz_zdr = {}
            self.Azimuth = np.zeros(bb.nbins)
            for j in xrange(len(self.Azimuth)):
                self.rain_refl[np.str(j)] = []
                self.drizz_zdr[np.str(j)] = []
                self.Azimuth[j] = bb.bin_width * j
        else:
            store = RawDataStorage(filename=self.save)
            self.Azimuth = store.Azimuth
            self.rain_refl = store.rain_refl
            self.drizz_zdr = store.drizz_zdr
            if bb.nbins != len(self.Azimuth):
                warn('Wrong number of azimuth bins, going to #FAIL')

    def get_statistics(self):
        """
        Method to compute blockage statistics as functions of azimuth. Focus
        is on median values along with 95% confidence intervals.
        """
        length = len(self.Azimuth)
        self.rain_stats = {}
        self.rain_stats['N'] = np.zeros(length)
        self.rain_stats['low'] = np.zeros(length)
        self.rain_stats['median'] = np.zeros(length)
        self.rain_stats['high'] = np.zeros(length)
        self.drizzle_stats = {}
        self.drizzle_stats['N'] = np.zeros(length)
        self.drizzle_stats['low'] = np.zeros(length)
        self.drizzle_stats['median'] = np.zeros(length)
        self.drizzle_stats['high'] = np.zeros(length)
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
        fileobj = open(self.image_dir+'blockage_data_kdp_method.txt', 'w')
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
        a single basestring (e.g., one file)
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
    and differential reflectivity data. Uses pickle and stores the saved
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
        with open(filename, 'wb') as f: 
            pickle.dump(self, f)

    def load_raw_data(self, filename):
        """
        Loads data from pickled file.
        """
        if filename is None:
            filename = './temporary_block_data.dat'
        with open(filename, 'rb') as f:
            loadobj = pickle.load(f)
        self.populate_attributes(loadobj)

    def populate_attributes(self, obj):
        """
        Only certain attributes are actually saved.
        """
        self.rain_refl = obj.rain_refl
        self.drizz_zdr = obj.drizz_zdr
        self.Azimuth = obj.Azimuth

#####################################

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
    return np.int32(np.round(index))

def calc_median_ci(array):
    """Fix later to actually get confidence interval for median, not mean"""
    if len(array) > 1:
        N = np.float(len(array))
        median = np.median(array)
        sigma = np.std(array)
        lo = median - 1.96 * sigma / np.sqrt(N)
        hi = median + 1.96 * sigma / np.sqrt(N)
        return len(array), lo, median, hi
    else:
        return len(array), BAD, BAD, BAD

#####################################

#####################################





