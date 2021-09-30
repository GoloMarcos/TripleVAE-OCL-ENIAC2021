#download Datasets
[ ! -f "../Events.zip" ] && echo "Downloading Events.zip" && gdown --id 1htCJ4b2rFiifPvEbo0GJpP1x8ZwCOS1v -O "../Events.zip"

[ ! -d "../Events" ] && echo "Unziping Events.zip" && unzip ../Events.zip -d ../