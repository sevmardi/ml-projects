python train.py \
--dataroot datasets \
--name test_recnt \
--model pix2pix \
--which_model_netG unet_128 \
--which_direction BtoA \
--lambda_A 100 \
--dataset_mode aligned \
--use_spp \
--no_lsgan \
--norm batch