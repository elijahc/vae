convert -delay 15 -loop 0 *.png recon_learning.gif
gifsicle -O3 --colors 256 recon_learning.gif -o recon_learning_optimized.gif
URL=cat recon_learning_optimized.gif | imgur-uploader
