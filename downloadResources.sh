#~/bin/sh

if [ ! -d "resources" ]; then
	echo "Creating resources directory"
	mkdir -p resources/img/sipi
	mkdir -p resources/calgary

	echo "Downloading SIPI misc images dataset"
	wget http://sipi.usc.edu/database/misc.tar.gz

	echo "Downloading Calgary corpus"
	wget http://corpus.canterbury.ac.nz/resources/calgary.tar.gz

	echo "Extracting archives"
	tar -xzf misc.tar.gz -C resources/img/sipi/
	tar -xzf calgary.tar.gz -C resources/calgary/
	
	echo "Cleaning up"
	mv misc.tar.gz resources/
	mv calgary.tar.gz resources/
fi
