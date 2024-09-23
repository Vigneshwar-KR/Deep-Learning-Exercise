URL="https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/${1}.tar.gz"
echo $URL
TAR_FILE=./$FILE.tar.gz
wget -O $TAR_FILE $URL
tar -zxvf $TAR_FILE -C ./
rm $TAR_FILE