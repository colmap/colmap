
# Building & running COLMAP using Docker

## Quick start:

1. Make sure docker is installed on your local machine.
2. Run the 'QUICK_START' script, using the *full local path* to your prefered local working directory (perhaps a folder with your input files/images, etc):

	./QUICK_START.sh /path/where/your/working/folder/is

3. This will put you in a directory (inside the Docker container) mounted to the local path you specified so that you can run COLMAP binaries on your own inputs. Enjoy!

## Build from scratch:

You can also build the docker image from scratch based on the **Dockerfile** (perhaps with your own changes / modifications) using:

	./buildFromScratch.sh /path/where/your/working/folder/is

## NOTES

+ This workflow is pinned to build from [COLMAP 3.6-dev.3](https://github.com/tjdahlke/colmap/releases/tag/3.6-dev.3.) To build from a different release, or build from the latest commit on master, open up the Dockerfile and comment/uncomment as directed.

+ Running COLMAP binaries can use a lot of memory (depending on the size of your data set/ imagery). Docker has a relatively small default memory setting (2Gb on Mac). You will probably want to increase this before you run any larger workflows. From Docker desktop on Mac for example, just open the Docker GUI, go to the *Advanced* tab and increase via the slider:

![alt text][dockerParam]

[dockerParam]: https://i.stack.imgur.com/6iWiW.png "Recommend increasing memory to >4Gb"
