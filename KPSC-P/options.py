import argparse
import os


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        #### train #####
        parser.add_argument('--now_best',             type=float,   default=100)
        parser.add_argument('--load_pretrained',      action='store_true')
        parser.add_argument('--load_model_path',      type=str,     default='../train_results/checkpoints/train_latest.pth')
        parser.add_argument('--start_epoch',          type=int,     default=0)
        parser.add_argument('--batchsize',            type=int,     default=112,            help="batchsize for training and testing")
        parser.add_argument('--workers',              type=int,     default=4,              help="thread number for read images") 
        parser.add_argument('--epochs',               type=int,     default=1000,           help="epochs for train process")
        parser.add_argument('--loadOnInit',           action='store_true',                  help='set true if memory is very large')
        parser.add_argument('--pin_memo',             action='store_true',                  help='decide if pin_memory is True')
        parser.add_argument('--learning_rate',        type=float,   default=1e-4,           help="learning rate for training")
        parser.add_argument('--clip_learning_rate',   type=float,   default=1e-5,           help="learning rate for training")
        parser.add_argument('--print_freq',           type=int,     default=10)
        parser.add_argument('--eval_freq',            type=int,     default=10)
                #### optim #####
        parser.add_argument('--optim',                type=str,     default='adam',         help="adam, sgd, adamw")
        parser.add_argument('--lr_sheduler',          type=str,     default='cosine')
        parser.add_argument('--lr_sheduler_per_epoch',type=int,     default=50)
        parser.add_argument('--lr_warmup_step',       type=int,     default=5)

        parser.add_argument('--frozen_clip',          action='store_true')
        
        ####backbone clip networks####
        parser.add_argument('--backbone_path',        type=str,     default='clip_code/checkpoints/ViT-B-32.pt')
        parser.add_argument('--input_size',           type=int,     default='224',                                                   help="size of input image")

        ####visual prompt####
        parser.add_argument('--tsm',                  action='store_true')
        parser.add_argument('--visual_drop_out',      type=float,   default=0.0)
        parser.add_argument('--visual_emb_dropout',   type=float,   default=0.0)
        parser.add_argument('--joint',                action='store_true')
        parser.add_argument('--sim_header',           type=str,     default='Transf',                                                help="Transf   meanP  LSTM Conv_1D Transf_cls")

        ####PromptLearner####
        parser.add_argument('--CTX_INIT',             type=str,     default='a photo of a',                                          help="use given words to initialize context vectors")
        parser.add_argument('--N_CTX',                type=int,     default=5,                                                      help="number of context words (tokens) in prompts") 
        parser.add_argument('--VERB_TOKEN_POSITION',  type=str,     default='middle',                                                help="'middle' or 'end' or 'front'")
        parser.add_argument('--n_verb',               type=int,     default=5,                                                      help="number of verbs words (tokens) in prompts") 

        

        ####Datasets####
        parser.add_argument('--dataset',              type=str,     default='ActivityNet')
        parser.add_argument('--datasets_root_path',   type=str,     default='/SSD2T/Datasets/Charades/raw_video/rgb_frame_24fps/Charades_v1_rgb')
        parser.add_argument('--train_json_file',      type=str,     default='proposal_generateion/psvl_anno/charades/charades_train_pseudo_supervision_TEP_PS.json')
        parser.add_argument('--num_segments',         type=int,     default=8,                                                       help="numbers of segemnets") 
        parser.add_argument('--new_length',           type=int,     default=1,                                                       help="per frame for a segements") 
        parser.add_argument('--image_tmpl',           type=str,     default='{}-{:06d}.jpg')
        #####1. transform#####
        parser.add_argument('--randaug_N',            type=int,     default=2)
        parser.add_argument('--randaug_M',            type=int,     default=9)
        parser.add_argument('--random_shift',         action='store_true')
        parser.add_argument('--index_bias',           type=int,     default=1)

        ##### DDP #####
        parser.add_argument("--local_rank",           default=-1,   type=int, help="distribted training")
        parser.add_argument('--is_distributed',       action='store_true')
        parser.add_argument('--port',                 type=int,     default=29582,                      help="ddp port")
        ##### save results #####
        ##### log #####
        parser.add_argument('--logdir',               type=str,     default="../train_results/log",            help="log directory") 
        parser.add_argument("--log_name",             type=str,     default="remo.log",                 help="The Path for Saving TF LOG")
        ##### model #####
        parser.add_argument('--modeldir',             type=str,     default="../train_results/checkpoints",     help="model directory")

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        return message

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        return self.gather_options()
