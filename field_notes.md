masking recon is strictly neccessary, masking original is not, but masking it (symmetry) helps massivley
sepeareting sigregs: sigreg_loss = sigreg_weight * (sigreg(m_z_flat) + sigreg(mrec_z_flat))
is essetial.

adding regular z to this sigreg didnt seem to help in anyway and may even have been less stable.