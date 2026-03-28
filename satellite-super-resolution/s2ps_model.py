"""
Multi-Band SR Models for BasicSR
==================================

Two models for satellite super-resolution:
1. MultiBandSRModel - For PSNR training (L1 + SAM loss)
2. MultiBandGANModel - For GAN fine-tuning (adds perceptual + adversarial)

Both handle multi-band (>3 channel) validation without RGB conversion errors.

YAML usage::

    model_type: MultiBandSRModel   # For PSNR training
    model_type: MultiBandGANModel  # For GAN fine-tuning
"""

import torch
from basicsr.models.sr_model import SRModel
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MultiBandSRModel(SRModel):
    """
    Extends SRModel for multi-band satellite imagery.

    Features:
    - Handles >3 band images in validation (no RGB conversion)
    - Adds SAM loss support for spectral fidelity
    - Works with MultiBandRRDBNet, MultiBandSwinIR, etc.
    """

    def __init__(self, opt):
        super().__init__(opt)

    def init_training_settings(self):
        """Extend parent to add SAM loss from config."""
        super().init_training_settings()

        # Build SAM loss if specified
        train_opt = self.opt.get('train', {})
        if 'sam_opt' in train_opt:
            from basicsr.utils.registry import LOSS_REGISTRY
            sam_opt = train_opt['sam_opt'].copy()
            sam_type = sam_opt.pop('type')
            self.cri_sam = LOSS_REGISTRY.get(sam_type)(**sam_opt).to(self.device)
        else:
            self.cri_sam = None

    def optimize_parameters(self, current_iter):
        """Add SAM loss to standard optimization."""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = {}

        # Pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (if configured, uses first 3 bands)
        if self.cri_perceptual:
            out_rgb = self.output[:, :3] if self.output.shape[1] > 3 else self.output
            gt_rgb = self.gt[:, :3] if self.gt.shape[1] > 3 else self.gt
            l_percep, l_style = self.cri_perceptual(out_rgb, gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # SAM loss (spectral angle)
        if self.cri_sam:
            l_sam = self.cri_sam(self.output, self.gt)
            l_total += l_sam
            loss_dict['l_sam'] = l_sam

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Override to handle multi-band images without RGB conversion."""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt['val']['metrics'].keys()
            }

        pbar = None
        if hasattr(dataloader.dataset, '__len__'):
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = f'val_{idx:04d}'
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']  # Keep as tensor, don't convert
            gt_img = visuals['gt'] if 'gt' in visuals else None

            # Skip image saving for multi-band (would fail with cv2)
            # Images can be visualized separately with proper band selection

            # Calculate metrics
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    from basicsr.metrics import calculate_metric
                    # Pass tensors directly to our custom metrics
                    if gt_img is not None:
                        metric_value = calculate_metric(
                            {'img': sr_img.cpu().numpy(),
                             'img2': gt_img.cpu().numpy()}, opt_
                        )
                        self.metric_results[name] += metric_value

            if pbar:
                pbar.update(1)
                pbar.set_description(f'{img_name}')

        if pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


@MODEL_REGISTRY.register()
class MultiBandGANModel(SRGANModel):
    """
    Extends SRGANModel for multi-band GAN training.

    Features:
    - BGR extraction for VGG-based perceptual loss
    - SAM loss for spectral fidelity
    - Handles multi-band validation
    """

    def __init__(self, opt):
        super().__init__(opt)
        self._rgb_indices = [0, 1, 2]  # First 3 bands for VGG

    def _extract_rgb(self, x):
        """Extract first 3 bands for VGG-based losses."""
        if x.shape[1] > 3:
            return x[:, self._rgb_indices]
        return x

    def init_training_settings(self):
        """Extend parent to add SAM loss."""
        super().init_training_settings()

        train_opt = self.opt.get('train', {})
        if 'sam_opt' in train_opt:
            from basicsr.utils.registry import LOSS_REGISTRY
            sam_opt = train_opt['sam_opt'].copy()
            sam_type = sam_opt.pop('type')
            self.cri_sam = LOSS_REGISTRY.get(sam_type)(**sam_opt).to(self.device)
        else:
            self.cri_sam = None

    def optimize_parameters(self, current_iter):
        """Override to inject RGB extraction for perceptual loss."""
        # --- Generator update ---
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = {}

        # Pixel loss (all bands)
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (RGB only for VGG)
        if self.cri_perceptual:
            out_rgb = self._extract_rgb(self.output)
            gt_rgb = self._extract_rgb(self.gt)
            l_percep, l_style = self.cri_perceptual(out_rgb, gt_rgb)
            if l_percep is not None:
                l_g_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style

        # GAN loss
        if hasattr(self, 'cri_gan') and self.cri_gan:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # SAM loss
        if hasattr(self, 'cri_sam') and self.cri_sam:
            l_sam = self.cri_sam(self.output, self.gt)
            l_g_total += l_sam
            loss_dict['l_sam'] = l_sam

        l_g_total.backward()
        self.optimizer_g.step()

        # --- Discriminator update ---
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()

        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()

        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Override to handle multi-band images without RGB conversion."""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt['val']['metrics'].keys()
            }

        pbar = None
        if hasattr(dataloader.dataset, '__len__'):
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = f'val_{idx:04d}'
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    from basicsr.metrics import calculate_metric
                    if gt_img is not None:
                        metric_value = calculate_metric(
                            {'img': sr_img.cpu().numpy(),
                             'img2': gt_img.cpu().numpy()}, opt_
                        )
                        self.metric_results[name] += metric_value

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
