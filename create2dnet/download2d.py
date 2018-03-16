import wget

lsp_url = 'http://sam.johnson.io/research/lsp_dataset.zip'
lspet_url = 'http://sam.johnson.io/research/lspet_dataset.zip'
mpii2d_image_url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
mpii2d_annot_url =  'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip'

lsp_filename = wget.download(lsp_url)
lspet_filename = wget.download(lspet_url)
mpii2d_image_filename = wget.download(mpii2d_image_url)
mpii2d_annot_filename = wget.download(mpii2d_annot_url)
