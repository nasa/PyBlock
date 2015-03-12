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
v1.0 - 03/12/2015

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
v1.0 Functionality (03/12/2015)
1.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import pyart
from warnings import warn
from csu_radartools import csu_kdp, csu_misc
import dualpol
from singledop import fn_timer

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
          'kd': None, 'ld': None, 'save': False, 'sounding': DEFAULT_SND,
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

class BeamBlockSingleVolume(object):

    """
    TO DO
    -----
    1. FSC method implementation
    2. Move stuff out of __init__()
    """

    @fn_timer
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
        self.save = kwargs['save']
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
        az = self.radar.azimuth['data']
        rng = self.radar.range['data'] / RANGE_MULT
        self.range, self.azimuth = np.meshgrid(rng, az)

    def get_bins(self, bin_width):
        """
        Assumes azimuth stays between 0 and 360 deg.
        """
        self.bin_width = bin_width
        self.nbins = get_index(360.0, bin_width)
        self.azimuth_indices = get_index(self.azimuth, bin_width)
        cond = self.azimuth_indices == self.nbins
        self.azimuth_indices[cond] = 0
    
    def get_range_mask(self, rng_thresh):
        """
        Think works if bin_width = 1.0 deg, but does it
        work if data are undersampled relative to bin size?
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
        dummy = []
        for i in xrange(self.nbins):
            dummy.append(rng_thresh)
        self.range_thresh = dummy

    def partition_rain_data(self, rain_kdp_thresh, rain_dz_thresh, rain_dp_thresh):
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
        self.drizzle_not_fhc = self.fhc != 1
        self.drizzle_not_kdp = np.logical_or(self.kd < drizzle_kdp_thresh[0],
                                             self.kd > drizzle_kdp_thresh[1])
        self.drizzle_not_dz = self.dz > drizzle_dz_thresh
        self.drizzle_good_mask = consolidate_masks([self.bad_data_mask,
             self.range_mask, self.drizzle_not_fhc, self.drizzle_not_kdp,
             self.drizzle_not_dz])
    
    def group_rain_data(self):
        indices = self.azimuth_indices[self.rain_good_mask]
        dzvals = self.dz[self.rain_good_mask]
        self.rain_binned_dz = []
        for i in xrange(self.nbins):
           self.rain_binned_dz.append(dzvals[indices == i])
    
    def group_drizzle_data(self):
        indices = self.azimuth_indices[self.drizzle_good_mask]
        drvals = self.dr[self.drizzle_good_mask]
        self.drizzle_binned_dr = []
        for i in xrange(self.nbins):
           self.drizzle_binned_dr.append(drvals[indices == i])
    
    def get_bad_data_mask(self, magnetron=False):
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

class BeamBlockMultiVolume(object):
    """
    TO DO
    -----
    1. Periodic file saving
    2. Better naming for images
    3. DR saving/images
    4. Move stuff out of __init__()
    5. Sounding processing
    6. FSC implementation
    """

    def __init__(self, files, image_dir='./', image_ext='.png', **kwargs):
    
        kwargs = dualpol.check_kwargs(kwargs, DEFAULT_KW)
        self.check_file_list(files)
        self.get_sounding_list(**kwargs)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.process_multiple_volumes(**kwargs)
    
    def process_multiple_volumes(self, **kwargs):
        for i, filen in enumerate(self.file_list):
            kwargs['sounding'] = self.sounding_list[i]
            bb = BeamBlockSingleVolume(filen, **kwargs)
            if i == 0:
                self.rain_refl = {}
                self.drizz_zdr = {}
                for j in xrange(bb.nbins):
                    self.rain_refl[np.str(j)] = []
                    self.drizz_zdr[np.str(j)] = []
            for j in xrange(bb.nbins):
                if len(bb.rain_binned_dz[j]) > 0:
                    self.rain_refl[np.str(j)] =\
                        np.append(self.rain_refl[np.str(j)],
                                  bb.rain_binned_dz[j])
                if len(bb.drizzle_binned_dr[j]) > 0:
                    self.drizz_zdr[np.str(j)] =\
                        np.append(self.drizz_zdr[np.str(j)],
                                  bb.drizzle_binned_dr[j])
            if i % 10 == 0:
                print 'i=', i, 'of', len(self.file_list)-1
                self.make_plots(bb)
                
    def make_plots(self, bb):
        #Make Rainfall Reflectivity Plots
        for j in xrange(bb.nbins):
            siz = len(self.rain_refl[np.str(j)])
            if siz > 0:
                plt.plot(np.zeros(siz)+j*bb.bin_width, self.rain_refl[np.str(j)],
                         'k.', ms=1)
                if len(self.rain_refl[np.str(j)]) > 1:
                    plt.plot([j*bb.bin_width],
                             np.median(self.rain_refl[np.str(j)]), 'bD')
        plt.xlim([0,360])
        plt.title('KDP Method - Rainfall')
        plt.xlim('Azimuth (deg)')
        plt.ylim('Reflectivity (dBZ)')
        plt.savefig(self.image_dir+'block_rain_kdp_method'+self.image_ext)
        plt.close()
        #Make Drizzle Differential Reflectivity Plots
        for j in xrange(bb.nbins):
            siz = len(self.drizz_zdr[np.str(j)])
            if siz > 0:
                plt.plot(np.zeros(siz)+j*bb.bin_width, self.drizz_zdr[np.str(j)],
                         'k.', ms=1)
                if len(self.drizz_zdr[np.str(j)]) > 1:
                    plt.plot([j*bb.bin_width],
                             np.median(self.drizz_zdr[np.str(j)]), 'bD')
        plt.xlim([0,360])
        plt.title('KDP Method - Drizzle')
        plt.xlim('Azimuth (deg)')
        plt.ylim('Differential Reflectivity (dB)')
        plt.savefig(self.image_dir+'block_drizzle_kdp_method'+self.image_ext)
        plt.close()

    def check_file_list(self, files):
        """Checks if argument is list or a single basestring"""
        if isinstance(files, basestring):
            self.file_list = [files]
        else:
            self.file_list = files

    def get_sounding_list(self, **kwargs):
        """ 
        User is responsible for inputing properly formatted string
        file names to program. Limited error checking is done.
        """
        sounding = kwargs['sounding']
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

#####################################

#####################################

def consolidate_masks(bad_masks):
    """
    bad_masks = list of bad data masks to apply to array
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
    index = azimuth / bin_width
    return np.int32(np.round(index))



#####################################

#####################################





