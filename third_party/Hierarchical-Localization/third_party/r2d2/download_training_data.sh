# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

CODE_ROOT=`pwd`
if [ ! -e data ]; then
    echo "Error: missing data/ folder"
    echo "First, create a folder that can host (at least) 15 GB of data."
    echo "Then, create a soft-link named 'data' that points to it."
    exit -1
fi

# download web images from the revisitop1m dataset
WEB_ROOT=data/revisitop1m
mkdir -p $WEB_ROOT
cd $WEB_ROOT
if [ ! -e 0d3 ]; then
    for i in {1..5}; do
        echo "Installing the web images dataset ($i/5)..."
        if [ ! -f revisitop1m.$i.tar.gz ]; then
            wget http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.$i.tar.gz
        fi
        tar -xzvf revisitop1m.$i.tar.gz
        rm -f revisitop1m.$i.tar.gz
    done
fi
cd $CODE_ROOT

# download aachen images
AACHEN_ROOT=data/aachen
mkdir -p $AACHEN_ROOT
cd $AACHEN_ROOT
if [ ! -e "images_upright" ]; then
    echo "Installing the Aachen dataset..."
    fname=database_and_query_images.zip
    if [ ! -f $fname ]; then
        echo "File not found: $fname"
        exit -1
    else
        unzip $fname
        rm -f $fname
    fi
fi

# download style transfer images
if [ ! -e "style_transfer" ]; then
    echo "Installing the Aachen style-transfer dataset..."
    fname=aachen_style_transfer.zip
    if [ ! -f $fname ]; then
        wget http://download.europe.naverlabs.com/3DVision/aachen_style_transfer.zip $fname
    fi
    unzip $fname
    rm -f $fname
fi

# download optical flow pairs
if [ ! -e "optical_flow" ]; then
    echo "Installing the Aachen optical flow dataset..."
    fname=aachen_optical_flow.zip
    if [ ! -f $fname ]; then
        wget http://download.europe.naverlabs.com/3DVision/aachen_optical_flow.zip $fname
    fi
    unzip $fname
    rm -f $fname
fi
cd $CODE_ROOT

echo "Done!"

