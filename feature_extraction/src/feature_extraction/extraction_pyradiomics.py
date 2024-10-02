import pandas as pd
from classrad.data.dataset import ImageDataset
from classrad.feature_extraction.extractor import FeatureExtractor
from pathlib import Path
import src.data_cleaning.config as config


def extract_features(paths_file: Path,
                     kernel_radius: int = config.KERNEL_RADIUS,
                     ):
    # sanity check
    # assert str(kernel_radius) in paths_file.stem

    recover_paths = False

    # default to non-dilated data
    if kernel_radius == 0:
        # paths_file = config.cfg.image_path_file
        recover_paths = False

    # Load table with paths
    df = pd.read_csv(paths_file)


    if recover_paths:
        # mk paths correct
        lvl = config.cfg.image_path_file_parent_level
        img_dir = config.cfg.image_dir
        df[['image_path', 'mask_path']] = df[['image_path', 'mask_path']].applymap(
            lambda x: (img_dir / Path(*Path(x).parts[-lvl:])).absolute().__str__())

    # check if all filepath exist
    exist = df[['image_path', 'mask_path']].applymap(
        lambda x: Path(x).is_file())
    # DEBUG: missing data
    df = df.loc(axis=0)[exist.values.all(axis=1)]
    assert exist.values.all()


    for relax_mode in ['T1', 'T2']:
        # generate Dataset
        image_dataset = ImageDataset.from_dataframe(
            df=df[df['study_type'].str.contains(relax_mode)],
            image_colname="image_path",
            mask_colname="mask_path",
            id_colname="ID",
        )
        # Feature extraction
        extraction_params = (config.cfg.pyradiomics_params_dir / '{}.yaml'.format(relax_mode)).__str__()
        feature_output_path = config.RESULT_DIR / "features_{}.csv".format(relax_mode)
        extractor = FeatureExtractor(
            image_dataset,
            out_path=feature_output_path,
            extraction_params=extraction_params,
            num_threads=config.cfg.n_cores
        )
        extractor.extract_features()
        features = pd.read_csv(feature_output_path, index_col=None)
        features[['image_path', 'mask_path']] = features[['image_path', 'mask_path']].applymap(
            lambda x: Path(x).relative_to(config.cfg.root_dir).__str__())
        features.to_csv(feature_output_path, index=False)

    return


def extract_dilated():
    for kernel_radius in config.KERNEL_RADIUSES:
        print('Starting Feature extraction for kernel_radius %i' % kernel_radius)
        extract_features(config.TABLE_DIR / 'dilated_paths_{}.csv'.format(kernel_radius), kernel_radius=kernel_radius)


if __name__ == "__main__":
    # parse args
    import argparse


    # helper function for boolean args
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Run Dilatation')
    parser.add_argument('--radius',
                        metavar='R',
                        type=int,
                        nargs='?',
                        default=3,
                        help='the kernel_radius for sitk.sitkBall as structure elementfor BinDilate (Default 3)')
    parser.add_argument('--jobarr',
                        type=str2bool,
                        nargs='?',
                        const=True, default=False,
                        help='To be used with jobarray; hence radius should be given as arg')
    args = parser.parse_args()
    kernel_radius = args.radius
    # print('Starting Feature extraction for kernel_radius %i' % kernel_radius)
    if not args.jobarr:
        print("--jobarr is False. Starting loop.")
        extract_dilated()
    else:
        print("--jobarr is True. Executing Feature extraction for %i." % kernel_radius)
        extract_features(config.TABLE_DIR / 'paths.csv'.format(kernel_radius), kernel_radius=kernel_radius)
