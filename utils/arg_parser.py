import argparse
import yaml

def nullable_string(val):
    if not val:
        return None
    return val

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument('--output_base', default='', type=str)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_suffix", default="")

    parser.add_argument("--model_type", default='ngp', type=str)
    parser.add_argument('--checkpoint_file', default='', type=nullable_string)

    parser.add_argument("--input_dir", default="")
    parser.add_argument("--input_file", default="")
    parser.add_argument('--dataset_name', default="", type=str)

    parser.add_argument("--n_steps", default=500, type=int, help="Number of training steps")
    parser.add_argument("--target_sample_batch_size", default=1<<20, type=int,
                        help='Number of sampled volumetric positions')
    parser.add_argument("--num_points", default=1024, type=int,
                        help='Number of sampled volumetric points per ray')
    parser.add_argument("--num_rays", type=int, default=1024)
    parser.add_argument("--num_proj", default=216, type=int)
    parser.add_argument("--n_neurons", default=64, type=int)
    parser.add_argument("--n_hidden_layers", default=2, type=int)
    parser.add_argument("--bias_free", action='store_true')
    parser.add_argument("--no-bias_free", action='store_false', dest='bias_free')
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--encoding_type", default='HashGrid', type=str)
    parser.add_argument("--n_features_per_level", default=2, type=int)
    parser.add_argument("--N_max", default=1024, type=int)
    parser.add_argument("--N_min", default=16, type=int)
    parser.add_argument("--n_levels", default=16, type=int)
    parser.add_argument("--log2_hashmap_size", default=21, type=int)
    parser.add_argument("--encoding_scale" ,default=5., type=float)
    parser.add_argument("--resolution", default=[128, 128, 128], type=int, nargs="+")

    parser.add_argument("--occupancy_threshold", default=1e-4, type=float)

    parser.add_argument("--regularization_mode", default='none', type=str)
    parser.add_argument("--reg_checkpoint", default='', type=str)
    parser.add_argument('--reg_n_iter', default=3, type=int)
    parser.add_argument("--red_denoising", action='store_true')
    parser.add_argument("--lambda_reg", default=0.1, type=float)
    parser.add_argument("--reg_patch_size", default=256, type=int)
    parser.add_argument("--reg_batch_size", default=64, type=int)
    parser.add_argument("--reg_start", default=0, type=int)
    parser.add_argument("--reg_sampling", default='cube', type=str)

    parser.add_argument("--test_chunk_size", default=1024, type=int)
    parser.add_argument("--log_name", default="train", type=str)
    
    #Dataset
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--center_crop", action='store_true')
    parser.add_argument("--axial_center_crop", action='store_true')
    
    # Model args
    parser.add_argument("--stem_size", default=5, type=int)
    parser.add_argument("--residual_learning", action='store_true')
    parser.add_argument("--residual_scale_factor", default=1.0, type=float)
    parser.add_argument("--downscaling_layer", default='strided_conv', type=str)
    parser.add_argument("--upscaling_layer", default='transposeconv_nogroup', type=str)
    parser.add_argument("--interpolation" ,default='bilinear', type=str)
    parser.add_argument("--augmentation", action='store_true')
    parser.add_argument("--spectral_normalization", action='store_true')
    parser.add_argument("--normalization_layer", default="LayerNorm", type=str)
    parser.add_argument("--normalization_bias_free", action='store_true')
    parser.add_argument("--final_activation", default="Identity", type=str)
    parser.add_argument("--legacy_head", action='store_true')
    
    parser.add_argument("--additional_input_channels", default=0, type=int)
    parser.add_argument("--input_memory", action='store_true')
    parser.add_argument("--iterative_model", default=-1, type=int)
    parser.add_argument("--jfb", action='store_true')
    parser.add_argument("--forward_iterates", action="store_true")
    
    parser.add_argument("--pnp", action='store_true')
    parser.add_argument("--pnp_sigma", default=10., type=float)
    parser.add_argument("--fixed_sigma", action='store_true')
    parser.add_argument("--noise_level", default=0.0, type=float)
    
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument("--ema_validation", action='store_true')
    
    # Optimizer args
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dampening', default=0., type=float)
    parser.add_argument('--nesterov', action='store_true')
    # parser.add_argument('--betas', default=[0.9, 0.999])
    parser.add_argument('--optimizer_eps', default=1e-8, type=float)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--weight_decay', default=4e-5, type=float)
    
    # Scheduler
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--init_lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--num_epochs_restart', default=-1, type=int)

    parser.add_argument('--num_warmup_epochs', default=5, type=int)
    parser.add_argument('--warmup_start', default=1e-8, type=float)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str)
    
    parser.add_argument("--acquisition_id", default='', type=str)
    parser.add_argument('--split_set', default='train', type=str)

    parser.add_argument('--inference_save', action='store_true')
    
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--no-pin_memory', dest='pin_memory', action='store_false')
    parser.add_argument('--memmap', action='store_true')
    parser.add_argument('--no-memmap', dest='memmap', action='store_false')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Enable automatic mixed precision training')

    parser.add_argument('--clip_grad', default=20., type=float) # TODO Legacy : should remove this
    parser.add_argument('--clip_grad_norm', default=0., type=float) # usual value : 20
    parser.add_argument('--clip_grad_value', default=0., type=float) # usual value : 5

    parser.add_argument("--cpu", action='store_true')
    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=10, type=int)

    return parser

def get_config_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    return config_parser

def parse_args():

    config_parser = get_config_parser()
    parser = get_arg_parser()

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text