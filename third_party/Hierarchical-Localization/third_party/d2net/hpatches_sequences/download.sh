#!/usr/bin/env bash

# Download the dataset
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

# Extract the dataset
tar xvzf hpatches-sequences-release.tar.gz

# Remove the high-resolution sequences
cd hpatches-sequences-release
rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
cd ..
