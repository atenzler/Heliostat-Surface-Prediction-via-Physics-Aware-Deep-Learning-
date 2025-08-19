"""
Parameter_Netzwerk.py

Configuration file for initializing and training DeepLARTS models.

This file collects all parameter dictionaries for:
- Architecture
- Encoders (convolutional, transformer-fusion, transformer-flux)
- StyleGAN integration
- Training & testing setup
- Data preprocessing

These arguments are passed into `my_deepLarts.init_deepLarts`.

Note:
- The overall architecture follows Lewen et al. (2024), (https://doi.org/10.48550/arXiv.2408.10802),
  with minor modifications introduced in this thesis.

"""


architecture_args = {"train_on_flux_density": False,
                     "use_transformer_encoder": False,
                     "final_activation": "leaky_relu",
                     "use_scalar_input": True,
                     "use_targetID": False,
                     "use_canting_vecs": False,
                     }


convolution_encoder_args = {"fmap_max": 256,
                            "network_capacity": 12,
                            "epsilon_AdaIN": 0.00001,
                            "pdropout_enc": 0.2,
                            "pdropout_styleMapping": 0.2,
                            "use_weight_demod": True,
                            "reduce_encoder_depth_by": 0,
                            "dense_mapping_to_latent": False,
                            "residuals": False,
                            }

transformer_fusion_encoder_args = {"dim": 128,
                                   "depth": 5,
                                   "heads": 5,
                                   "mlp_dim": 128,
                                   "patch_size": 16,
                                   "dropout": 0.2,
                                   "emb_dropout": 0.2
                                   }

transformer_flux_encoder_args = {"dim": 128,
                                 "depth": 5,
                                 "heads": 5,
                                 "mlp_dim": 128,
                                 "patch_size": 16,
                                 "dropout": 0.2,
                                 "emb_dropout": 0.2
                                 }

styleGAN_args = {"freeze_styleGAN": False,
                 "new_styleGAN": False,
                 }

training_args = {"training_loss": "MAE",  # L2, MAE, ACC
                 "give_flux_loss_every": 10,
                 "nepochs": 50,
                 "batch_size": 64,
                 "learning_rate": 1e-3,
                 "loss_multiplyier": 1e6,
                 "weight_decay": 10e-8,
                 "geometry_model": 'mean',

                 "train_on_flux_density": architecture_args["train_on_flux_density"],

                 "reduce_flux": True,
                 "p_reduce_flux": 0.5,

                 "apply_domain_randomization": False,

                 "jitter_helpos": True,
                 "p_jitter_helpos": 0.5,
                 "para_jitter_helpos": 0.001,

                 "jitter_sunpos": True,
                 "p_jitter_sunpos": 0.5,
                 "para_jitter_sunpos": 0.0001,

                 "apply_bilinear_smoothing": True,

                 "apply_gaussian_kernel_smoothing": False,

                 "apply_lowpass": False,
                 "p_apply_lowpass": 0.5,
                 "para_lowpass": 5,

                 "apply_surface_noise": True,
                 "p_apply_surface_noise": 0.5,
                 "para_surface_noise": 1e-6,

                 "apply_flux_noise": True,
                 "p_apply_flux_noise": 5,
                 "p_apply_flux_noise_channel_wise": 0.5,
                 "para_flux_noise": 0.05,  # absolute value

                 "apply_overexposure": True,
                 "p_apply_overexposure": 0.5,
                 "overexposure_clamp": 0.9,

                 "crop_flux": True,
                 "p_crop_flux": 0.5,

                 "rand_contrast": True,
                 "p_rand_contrast": 0.5,
                 "max_scale_rand_contrast": 0.1,

                 "add_vertical_stripes": True,
                 "n_vertical_stripes": 5,
                 "amp_start": 0.5,
                 "amp_end": 1.3,
                 "height_range": [1, 4],
                 "width_range": [6, 60],

                 "add_empty_target_images_and_unet": False,

                 #"finetuning_args": finetuning_args
                 }

test_args = {#"name_deepLarts_list": name_deepLarts_list,
             #"load_from_deepLarts_list": load_from_deepLarts_list,
             #"compare_models": compare_test_results_models,
             #"save_compared_models": save_compared_models,
             #"save_name_compared_models": save_name_compared_models,
             "testhel": "AF31",
             "n_target_fluxes_valid": 3,

             #"test_real": TEST,
             "test_on_statedic": ["best_MPE_valid_real",
                                  "best_flux_valid_real2unet"],

             "test_best_acc": True,
             #"test_in_and_extrapolation": APPLY,
             "test_on_MFT": True,
             "test_on_1_trg_flux": False,

             "test_sim": False,
             #"test_extended_statistics": test_extended_statistics_,
             #"predictions_for_best_acc_dataset": predictions_for_best_acc_dataset
             }

data_args = {"fluxes_subscript": "train",  # traincropped, train_unet
             "normalize_flux_sum_to_1": False,
             "normalize_flux_max_to_1": True,
             "appl_low_passfilter_on_data": False,
             "apply_bilinear_smoothing": not training_args["apply_domain_randomization"],
             "train_on_augmented_data": True}

