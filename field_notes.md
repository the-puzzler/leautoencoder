masking recon is strictly neccessary, masking original is not, but masking it (symmetry) helps massivley
sepeareting sigregs: sigreg_loss = sigreg_weight * (sigreg(m_z_flat) + sigreg(mrec_z_flat))
is essetial.

adding regular z to this sigreg didnt seem to help in anyway and may even have been less stable.

on synthetic data, pixel masking actually seems to be much better, even than patch


on cifar, pixel 70% got me reocnstrucitons that looked very washed out, mostly just shadows and lightning, clearly reconisable but not good recons. this probably means masking needs to change.--> either more msking or differnt kind.

patch masking at 70% introduced some strange checkerboard artefacts

pixel at 90% looked very similar to 70%, still more or less just bright and dark

70% channel mask seems to have many artifcats like patch did bt slightly different.


combinging masks like this:
pixel_mask = make_pixel_mask(images, mask_ratio=mask_ratio)
            channel_mask = make_channel_mask(images, mask_ratio=mask_ratio)
            mask = pixel_mask * channel_mask
at 70% on both is giving best result so far. hazy colour in right place. still mostly light and dark but still.

seperaated the masks now instead: 
mse_loss = (
    F.mse_loss(pixel_z_flat, pixel_rec_z_flat) +
    F.mse_loss(channel_z_flat, channel_rec_z_flat)
)

this actually reintroduced the artifacts that were on channel masking mode alone. interesting. maybe the improbmenet were from compounded masking (0.7*0.7)?




just did a 0.99 pixel mask run with the 656,131 model of the current commit, and the quality is excelent the problem is.... everything is snot coloured. i would say the detial is on par or better than the baseline recon except for colour.


crops are the way! they seem to fix all problems. see the math md explanation.


seems like batchnorm causes some instability later in training. probably becuase it was also being updated on crop branches, so removing that and testing again, should hopefully bring test in line with train.


just realised that data is normed to -1 -> 1 yet model output was constrained 0-1.... idiot.