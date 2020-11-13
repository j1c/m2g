#!/usr/bin/env python

"""
m2g.track
~~~~~~~~~~

Contains m2g's fiber reconstruction and tractography functionality.
Theory described here: https://neurodata.io/talks/ndmg.pdf#page=21
"""

# system imports
import os
from copy import deepcopy

# external package imports
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed

# dipy imports
from dipy.tracking.streamline import Streamlines
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.local_tracking import ParticleFilteringTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.stopping_criterion import CmcStoppingCriterion

from dipy.reconst.dti import fractional_anisotropy, TensorModel
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, recursive_response

from dipy.data import get_sphere
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter
from m2g.utils.gen_utils import timer

from m2g.stats import qa_tensor


def build_seed_list(mask_img_file, stream_affine, dens):
    """uses dipy tractography utilities in order to create a seed list for tractography

    Parameters
    ----------
    mask_img_file : str
        path to mask of area to generate seeds for
    stream_affine : ndarray
        4x4 array with 1s diagonally and 0s everywhere else
    dens : int
        seed density

    Returns
    -------
    ndarray
        locations for the seeds
    """

    mask_img = nib.load(mask_img_file)
    mask_img_data = mask_img.get_data().astype("bool")
    seeds = utils.random_seeds_from_mask(
        mask_img_data,
        affine=stream_affine,
        seeds_count=int(dens),
        seed_count_per_voxel=True,
    )
    return seeds


def tens_mod_fa_est(gtab, dwi_file, B0_mask):
    """Estimate a tensor FA image to use for registrations using dipy functions

    Parameters
    ----------
    gtab : GradientTable
        gradient table created from bval and bvec file
    dwi_file : str
        Path to eddy-corrected and RAS reoriented dwi image
    B0_mask : str
        Path to nodif B0 mask (averaged b0 mask)

    Returns
    -------
    str
        Path to tensor_fa image file
    """

    data = nib.load(dwi_file).get_fdata()

    print("Generating simple tensor FA image to use for registrations...")
    nodif_B0_img = nib.load(B0_mask)
    B0_mask_data = nodif_B0_img.get_fdata().astype("bool")
    nodif_B0_affine = nodif_B0_img.affine
    model = TensorModel(gtab)
    mod = model.fit(data, B0_mask_data)
    FA = fractional_anisotropy(mod.evals)
    FA[np.isnan(FA)] = 0
    fa_img = nib.Nifti1Image(FA.astype(np.float32), nodif_B0_affine)
    fa_path = f"{os.path.dirname(B0_mask)}/tensor_fa.nii.gz"
    nib.save(fa_img, fa_path)
    return fa_path


class RunTrack:
    def __init__(
        self,
        dwi_in,
        nodif_B0_mask,
        gm_in_dwi,
        vent_csf_in_dwi,
        csf_in_dwi,
        wm_in_dwi,
        gtab,
        mod_type,
        track_type,
        mod_func,
        qa_tensor_out,
        seeds,
        stream_affine,
        n_cpus,
    ):
        """A class for deterministic tractography in native space

        Parameters
        ----------
        dwi_in : str
            path to the input dwi image to perform tractography on.
            Should be a nifti, gzipped nifti, or other image that nibabel
            is capable of reading, with data as a 4D object.
        nodif_B0_mask : str
            path to the mask of the b0 mean volume. Should be a nifti,
            gzipped nifti, or other image file that nibabel is capable of
            reading, with data as a 3D object.
        gm_in_dwi : str
            Path to gray matter segmentation in EPI space. Should be a nifti,
            gzipped nifti, or other image file that nibabel is capable of
            reading, with data as a 3D object
        vent_csf_in_dwi : str
            Ventricular CSF Mask in EPI space. Should be a nifti,
            gzipped nifti, or other image file that nibabel is capable of
            reading, with data as a 3D object
        csf_in_dwi : str
            Path to CSF mask in EPI space. Should be a nifti, gzipped nifti, or other image file that nibabel compatable
        wm_in_dwi : str
            Path to white matter probabilities in EPI space. Should be a nifti,
            gzipped nifti, or other image file that nibabel is capable of
            reading, with data as a 3D object.
        gtab : gradient table
            gradient table created from bval and bvec files
        mod_type : str
            Determinstic (det) or probabilistic (prob) tracking
        track_type : str
            Tracking approach: local or particle
        mod_func : str
            Diffusion model: csd or csa
        qa_tensor: str
            path to store the qa for tensor/directions of model 
        seeds : ndarray
            ndarray of seeds for tractography
        stream_affine : ndarray
            4x4 2D array with 1s diagonaly and 0s everywhere else
        n_cpus : int, optional
            Number of cpus to use for computing peaks and tracks.
        """

        self.dwi = dwi_in
        self.nodif_B0_mask = nodif_B0_mask
        self.gm_in_dwi = gm_in_dwi
        self.vent_csf_in_dwi = vent_csf_in_dwi
        self.csf_in_dwi = csf_in_dwi
        self.wm_in_dwi = wm_in_dwi
        self.gtab = gtab
        self.mod_type = mod_type
        self.track_type = track_type
        self.qa_tensor_out = qa_tensor_out
        self.seeds = seeds
        self.mod_func = mod_func
        self.stream_affine = stream_affine
        self.n_cpus = int(n_cpus)

    @timer
    def run(self):
        """Creates the tracktography tracks using dipy commands and the specified tracking type and approach

        Returns
        -------
        ArraySequence
            contains the tractography track raw data for further analysis

        Raises
        ------
        ValueError
            Raised when no seeds are supplied or no valid seeds were found in white-matter interface
        ValueError
            Raised when no seeds are supplied or no valid seeds were found in white-matter interface
        """
        # Prep tracking
        self.prep_tracking()

        if self.mod_func == "csa":
            self.mod = self.odf_mod_est()
        elif self.mod_func == "csd":
            self.mod = self.csd_mod_est()
        if self.track_type == "local":
            tracks = self.local_tracking()
        elif self.track_type == "particle":
            tracks = self.particle_tracking()
        else:
            raise ValueError(
                "Error: Either no seeds supplied, or no valid seeds found in white-matter interface"
            )

        return tracks

    @staticmethod
    def make_hdr(streamlines, hdr):
        trk_hdr = nib.streamlines.trk.TrkFile.create_empty_header()
        trk_hdr["hdr_size"] = 1000
        trk_hdr["dimensions"] = hdr["dim"][1:4].astype("float32")
        trk_hdr["voxel_sizes"] = hdr["pixdim"][1:4]
        trk_hdr["voxel_to_rasmm"] = np.eye(4)
        trk_hdr["voxel_order"] = "RAS"
        trk_hdr["pad2"] = "RAS"
        trk_hdr["image_orientation_patient"] = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).astype("float32")
        trk_hdr["endianness"] = "<"
        trk_hdr["_offset_data"] = 1000
        trk_hdr["nb_streamlines"] = streamlines.total_nb_rows

        return trk_hdr

    def prep_tracking(self):
        """Uses nibabel and dipy functions in order to load the grey matter, white matter, and csf masks
        and use a tissue classifier (act, cmc, or binary) on the include/exclude maps to make a tissueclassifier object

        Returns
        -------
        ActStoppingCriterion, CmcStoppingCriterion, or BinaryStoppingCriterion
            The resulting tissue classifier object, depending on which method you use (currently only does act)
        """
        self.dwi_img = nib.load(self.dwi)
        self.data = self.dwi_img.get_data()

        # Load tissue maps and prepare tissue classifier
        self.gm_mask = nib.load(self.gm_in_dwi)
        self.gm_mask_data = self.gm_mask.get_data()
        self.wm_mask = nib.load(self.wm_in_dwi)
        self.wm_mask_data = self.wm_mask.get_data()
        self.wm_in_dwi_data = nib.load(self.wm_in_dwi).get_data().astype("bool")

        if self.track_type == "local":
            self.tiss_classifier_kwargs = dict(mask=self.wm_in_dwi_data)
        elif self.track_type == "particle":
            self.vent_csf_in_dwi = nib.load(self.vent_csf_in_dwi)
            self.vent_csf_in_dwi_data = self.vent_csf_in_dwi.get_data()
            voxel_size = np.average(self.wm_mask.get_header()["pixdim"][1:4])
            step_size = 0.2

            self.tiss_classifier_kwargs = dict(
                wm_map=self.wm_mask_data,
                gm_map=self.gm_mask_data,
                csf_map=self.vent_csf_in_dwi_data,
                step_size=step_size,
                average_voxel_size=voxel_size,
            )
        else:
            pass

    @timer
    def odf_mod_est(self):

        print("Fitting CSA ODF model...")
        self.mod = CsaOdfModel(self.gtab, sh_order=6)
        return self.mod

    @timer
    def csd_mod_est(self):

        print("Fitting CSD model...")
        try:
            print("Attempting to use spherical harmonic basis first...")
            self.mod = ConstrainedSphericalDeconvModel(self.gtab, None, sh_order=6)
        except:
            print("Falling back to estimating recursive response...")
            self.response = recursive_response(
                self.gtab,
                self.data,
                mask=self.wm_in_dwi_data,
                sh_order=6,
                peak_thr=0.01,
                init_fa=0.08,
                init_trace=0.0021,
                iter=8,
                convergence=0.001,
                parallel=False
            )
            print("CSD Reponse: " + str(self.response))
            self.mod = ConstrainedSphericalDeconvModel(
                self.gtab, self.response, sh_order=6
            )
        return self.mod

    @timer
    def local_tracking(self):
        # Common arguments for local tracking
        # Seeds are added later for multiprocessing support
        tracking_kwargs = dict(
            affine=self.stream_affine,
            step_size=0.5,
            return_all=True,
        )

        self.sphere = get_sphere("repulsion724")
        if self.mod_type == "det":
            print("Obtaining peaks from model...")
            self.mod_peaks = peaks_from_model(
                self.mod,
                self.data,
                self.sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=self.wm_in_dwi_data,
                npeaks=5,
                normalize_peaks=True,
                parallel=False
            )
            qa_tensor.create_qa_figure(
                self.mod_peaks.peak_dirs,
                self.mod_peaks.peak_values,
                self.qa_tensor_out,
                self.mod_func,
            )

            # Update kwargs dict
            tracking_kwargs["direction_getter"] = self.mod_peaks

        elif self.mod_type == "prob":
            print("Preparing probabilistic tracking...")
            print("Fitting model to data...")
            self.mod_fit = self.mod.fit(self.data, self.wm_in_dwi_data)
            print("Building direction-getter...")
            self.mod_peaks = peaks_from_model(
                self.mod,
                self.data,
                self.sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=self.wm_in_dwi_data,
                npeaks=5,
                normalize_peaks=True,
                parallel=False
            )
            qa_tensor.create_qa_figure(
                self.mod_peaks.peak_dirs,
                self.mod_peaks.peak_values,
                self.qa_tensor_out,
                self.mod_func,
            )
            try:
                print(
                    "Proceeding using spherical harmonic coefficient from model estimation..."
                )
                self.pdg = ProbabilisticDirectionGetter.from_shcoeff(
                    self.mod_fit.shm_coeff, max_angle=60.0, sphere=self.sphere
                )
            except:
                print("Proceeding using FOD PMF from model estimation...")
                self.fod = self.mod_fit.odf(self.sphere)
                self.pmf = self.fod.clip(min=0)
                self.pdg = ProbabilisticDirectionGetter.from_pmf(
                    self.pmf, max_angle=60.0, sphere=self.sphere
                )

            tracking_kwargs["direction_getter"] = self.pdg

        print("Reconstructing tractogram streamlines...")
        # Start parallel streamline generation
        streams = Parallel(self.n_cpus)(delayed(worker)(
            seeds=self.seeds[i::self.n_cpus],
            track_type='local',
            tiss_classifier_kwargs=self.tiss_classifier_kwargs,
            tracking_kwargs=tracking_kwargs
        ) for i in range(self.n_cpus))

        # Concatenate streamlines
        self.streamlines = Streamlines()
        for stream in streams:
            self.streamlines.extend(stream)

        return self.streamlines

    @timer
    def particle_tracking(self):
        # Common arguments for particle tracking
        # Seeds are added later for multiprocessing support
        tracking_kwargs = dict(
            affine=self.stream_affine,
            step_size=0.5,
            maxlen=1000,
            pft_back_tracking_dist=2,
            pft_front_tracking_dist=1,
            particle_count=15,
            return_all=True,
        )

        self.sphere = get_sphere("repulsion724")
        if self.mod_type == "det":
            tracking_kwargs["maxcrossing"] = 1

            print("Obtaining peaks from model...")
            self.mod_peaks = peaks_from_model(
                self.mod,
                self.data,
                self.sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=self.wm_in_dwi_data,
                npeaks=5,
                normalize_peaks=True,
                parallel=False
            )
            qa_tensor.create_qa_figure(
                self.mod_peaks.peak_dirs,
                self.mod_peaks.peak_values,
                self.qa_tensor_out,
                self.mod_func,
            )

            tracking_kwargs["direction_getter"] = self.mod_peaks

        elif self.mod_type == "prob":
            tracking_kwargs["maxcrossing"] = 2

            print("Preparing probabilistic tracking...")
            print("Fitting model to data...")
            self.mod_fit = self.mod.fit(self.data, self.wm_in_dwi_data)
            print("Building direction-getter...")
            self.mod_peaks = peaks_from_model(
                self.mod,
                self.data,
                self.sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
                mask=self.wm_in_dwi_data,
                npeaks=5,
                normalize_peaks=True,
                parallel=False
            )
            qa_tensor.create_qa_figure(
                self.mod_peaks.peak_dirs,
                self.mod_peaks.peak_values,
                self.qa_tensor_out,
                self.mod_func,
            )
            try:
                print(
                    "Proceeding using spherical harmonic coefficient from model estimation..."
                )
                self.pdg = ProbabilisticDirectionGetter.from_shcoeff(
                    self.mod_fit.shm_coeff, max_angle=60.0, sphere=self.sphere
                )
            except:
                print("Proceeding using FOD PMF from model estimation...")
                self.fod = self.mod_fit.odf(self.sphere)
                self.pmf = self.fod.clip(min=0)
                self.pdg = ProbabilisticDirectionGetter.from_pmf(
                    self.pmf, max_angle=60.0, sphere=self.sphere
                )

            tracking_kwargs["direction_getter"] = self.pdg

        print("Reconstructing tractogram streamlines...")
        # Start parallel streamline generation
        streams = Parallel(self.n_cpus)(delayed(worker)(
            seeds=self.seeds[i::self.n_cpus],
            track_type='particle',
            tiss_classifier_kwargs=self.tiss_classifier_kwargs,
            tracking_kwargs=tracking_kwargs
        ) for i in range(self.n_cpus))

        # Concatenate streamlines
        self.streamlines = Streamlines()
        for stream in streams:
            self.streamlines.extend(stream)

        return self.streamlines


def worker(seeds, track_type, tiss_classifier_kwargs, tracking_kwargs):
    if track_type == 'local':
        tracker = LocalTracking
        stopping_criterion = BinaryStoppingCriterion(**tiss_classifier_kwargs)
    elif track_type == 'particle':
        tracker = ParticleFilteringTracking
        stopping_criterion = CmcStoppingCriterion(**tiss_classifier_kwargs)

    # Needs copy for parallelization
    tracking_kwargs["direction_getter"] = deepcopy(tracking_kwargs["direction_getter"])
    tracking_kwargs["stopping_criterion"] = stopping_criterion

    # Initialization of tracking. The computation happens in the next step.
    streamlines_generator = tracker(seeds=seeds, **tracking_kwargs)

    # Generate streamlines object
    streamlines = Streamlines(streamlines_generator)

    # Filter out short tracks
    filtered_streamlines = [track for track in streamlines if len(track) > 60]

    return filtered_streamlines
