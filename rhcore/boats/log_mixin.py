from rhcore.utils.build_components import build_module, build_modules
from rhcore.loggers.helpers.base_image_visualizer import BaseImageVisualizer

class LogMixIn:

    def __init__(self, log_config_dict, viz_config_dict):
        self.loggers = build_modules(log_config_dict['loggers'])

        first_logger = next(iter(self.loggers.values()))

        if viz_config_dict.get('save_images', False):
            self.viz = BaseImageVisualizer(first_logger, 
                                            wnb=viz_config_dict.get('wnb', (0.5, 0.5)), 
                                            max_images=viz_config_dict.get('max_images', 4),
                                            dataformats=viz_config_dict.get('dataformats', 'CHW'))

        self.viz_config_dict = viz_config_dict
        self.log_config_dict = log_config_dict
    
    def take_a_log(self, data_dict, prefix):
        for item_name, item_value in data_dict.items():
            for _, logger in self.loggers.items():
                logger.log_metrics({item_name: item_value}, 
                                   step=self.get_global_step(), 
                                   prefix=prefix)

    def save_images(self, named_imgs, batch_idx):

        if self.viz is None:
            return

        """Visualize validation results."""
        if self.viz_config_dict.get('first_batch_only', True) and batch_idx == 0:
            # Limit the number of samples to visualize
            for key in named_imgs.keys():
                if named_imgs[key].shape[0] > self.viz_config_dict.get('num_vis_samples', 4):
                    named_imgs[key] = named_imgs[key][:self.viz_config_dict.get('num_vis_samples')]
            
            # Log visualizations to the experiment tracker
            self.viz(
                images_dict=named_imgs,
                keys=list(named_imgs.keys()),
                global_step=self.get_global_step(),
                prefix='val',
                texts='texts',
            )      

