from pathlib import Path
import multiprocessing
import os


class Configuration:
    def __init__(self):
        # Image Info
        self.view = 'sax'
        self.mode = 'lowres'  # or ['lowres', 'highres', 'map', 'MOCO_highres', 'MOCO_lowres', 'MOCO']
        self.t1t2 = 'T1'
        self.relaxmode = self.t1t2 + '_' + self.mode

        self.xy_dims_lut = {
            'T1_lowres': (164, 192),
            'T1_highres': (218, 256),
            'T2_map': (154, 192),
            'T1_MOCO_lowres': (164, 192),
            'T1_MOCO_highres': (218, 256),
            'T2_MOCO': (154, 192)
        }
        self.xy_dims = self.xy_dims_lut[self.relaxmode]
        if self.view == '4cv':
            self.z_dim = 1
        else:
            self.z_dim = 3

        # root_dir
        self.original_root_dir = Path(r'imdata\\')
        self.root_dir = self.original_root_dir

        # place where the contours to be found
        self.contourpath = self.root_dir / 'ship_data/contours/Konturen/AP'  # r"E:\Konturen"

        # place where the dicom files live
        self.dcmpath = self.root_dir / 'data/Mapping_Sequenzen'

        # place where the extracted data should live
        self.name = self.relaxmode + '_' + self.view
        self.outputfolder = 'transformed_data'

        self.savingpath = self.root_dir / self.outputfolder / "data_proper_nodirection_duplicated" / self.name
        self.pklpath = self.root_dir / self.outputfolder / "pklpath" / self.name

        # pathes for import
        self.root_dir = Path(r'/home/data/Mapping/transformed_data')
        self.segmentation_dir = self.root_dir / 'data_segmentiert'
        self.img_dir = self.root_dir
        self.img_sub_dirs = ['data_proper_nodirection_duplicated', 'data_proper_nodirection', 'data_proper']
        self.image_path_file_parent_level = 2
        self.image_path_file = self.root_dir / 'paths.csv'
        self.pyradiomics_params = self.root_dir / 'pyradiomics_params' / 'Maps_Cardiac_MRI.yaml'
        self.pyradiomics_params_dir = self.root_dir / 'pyradiomics_params'

        # path for export
        self.data_dir = self.root_dir / 'Radiomics'
        self.segmentation_out_dir = self.data_dir / 'data' / 'masks'
        self.img_out_dir = self.data_dir / 'data' / 'imgs'
        self.path_out_file = self.data_dir / 'tables' / 'paths.csv'

        # computation specification
        try:
            self.n_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
            print("No of CPU cores %i" % self.n_cores)
        except KeyError:
            self.n_cores = multiprocessing.cpu_count()
            print("No of CPU cores %i" % self.n_cores)
        except Exception:
            self.n_cores = 10
            print('Defaulting to 10 cores')

        self.n_jobs = self.n_cores - 2
        self.n_rep_jobs = 1
        self.n_total_jobs = self.n_cores - 1
        self.blocksize = 2000  # debug only limited number of files

        # debug:
        self.n_jobs = self.n_cores
        self.n_total_jobs = 1

    def set_dirs(self, relaxmode: str = None, view: str = None):
        if relaxmode is not None:
            self.relaxmode = relaxmode

        if view is not None:
            self.view = view

        # set dimensions accordingly
        self.xy_dims = self.xy_dims_lut[self.relaxmode]
        if self.view == '4cv':
            self.z_dim = 1
        else:
            self.z_dim = 3

        # place where the extracted data should live
        self.name = self.relaxmode + '_' + self.view
        self.outputfolder = 'transformed_data'
        self.savingpath = self.original_root_dir / self.outputfolder / "data_proper_nodirection_duplicated" / self.name
        self.pklpath = self.original_root_dir / self.outputfolder / "pklpath" / self.name

        return relaxmode, view
