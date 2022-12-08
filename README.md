Support my efforts: https://www.buymeacoffee.com/tinylucas
# Tacotron_Prosody_multi_GPU
This repo is the implementation of Tacotron-2 model for Korean voice.

#### For someone (like me) wanna check demo before go futher, I posted synthesized files in `samples` folder.


## Before starting:


### 0 - Requirement.txt install
There are some requirement must be met to install requirements.txt some of them cannot be install via pip command:
 > such as pyaudio: download here https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio and use `pip install *.whl` to install for windows, for linux: 
              `sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
              sudo apt-get install ffmpeg libav-tools
              sudo pip install pyaudio`
 > Or such as pytorch: download here https://pytorch.org/ and install as home page instruction...

### 1 - Download traindata from: https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
The downloaded folder has 4 subfolder named from 1~4. For simplicity, I remove subfolder and put all file in a folder called "wavs" - take a look at `Train_data_folder_structure.png`
And in case of the dataset author changes the file format, I have captured the screenshoot of `transcript.txt` file format as `transcript_file_format.png`
### 2 - Download pretrained model: [will be upload and update soon]
put downloaded pretrained model as structure as in `pretrained_folder_structure.PNG`

### 3 - install requirements: 
open terminal (or cmd interface in Windows), activate virtual environment you wanna use, or use your system environment. If you do not know how to do, take a cup of coffee and visit https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
there are some packages will require some other apps or libraries (such as pyaudio), or could not installed using pip (such as pytorch), you need using Google to find out how to install them.

### 4 - update slash symbol suitable for your OS: 

In case you are using different OS from me, go to `Utils\AudioPreprocess\AudioPreprocess.py` file, edit the line `349` like this:

		# folder_id = input_dir.split('/')[-1]  ### get speaker folder name (in linux)
		folder_id = input_dir.split('\\')[-1]  ### get speaker folder name (in windows)

##Start Preprocessing process:

open terminal (cmd) run: `python Utils\AudioProcessing\AudioPreprocess.py` --> this step will take input audio, and generate mels spectrogram, wav file, as well as linear spectrogram and put them in `Tacotron_input` folder.
This should take less than 2 minutes.

Preprocessing and pretrained model need sharing same hyper parameters, so, if you want to use pretrained model, do not change `hparams.py` (of course you can change some, but this will need deep understanding, not recommend) file in `Utils` folder.

## Start training process:

after preprocessing stage, run `python TacotronModel\train.py` to start training with default parameters, or add -h param to see parameters info

This will continuosly train model, using pretrained model, and after checkpoint_interval steps, it will automatically save in pretrained folder.
If you want train model from scratch, use `restore` parameter, or just delete pretrained model folder.


## Start synthesizing process:

with a well trained model, using `python TacotronModel\synthesize.py` command to synthesize. This step will take data from `Tacotron_input` folder, predict and save result in `tacotron_output` folder.
The output of synthesizing process, will be input of Wavenet vocoder model training stage. 

## Start Inferencing process.

You can totally synthesize audio without Wavenet vocoder (may be the difference is about audio quality). using: `python TacotronModel\synthesize.py --mode=inference`
This will generate output using texts (from `sentences.txt`) and save in `tacotron_output/inference`, the output audio, plot are in `tacotron_output/log-inference folder`. (as synthesizing process, the output in `tacotron_output/inference` folder will be input of Wavenet vocoder inference process.

Because we are working on Flowavenet model for realtime synthesis, I will not post how to train and use Wavenet  model here

Waiting for Flowavenet.






