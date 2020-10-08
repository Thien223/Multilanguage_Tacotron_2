import argparse

from distributed import apply_gradient_allreduce
import time

from numpy import finfo
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence, sequence_to_text
from waveglow.denoiser import Denoiser
from scipy.io import wavfile

hparams=create_hparams()
torch.manual_seed(hparams.seed)

def clean_text(txt: str) -> list:
	start_time = time.time()
	# splitter = SentenceSplitter(api=API.HNN)
	# paragraph = splitter(txt)
	# return paragraph
	txt_list = []
	import string
	max_len = 60
	s=txt
	txt_ = s.translate(str.maketrans('', '', string.punctuation.replace(',', '').replace('.', '').replace('-', '').replace('/', '')))
	txt_ = txt_.strip()

	while True:
		if ',,' in txt_:
			txt_ = txt_.replace(',,', ',')
		else:
			break

	while True:
		if '..' in txt_:
			txt_ = txt_.replace('..', '.')
		else:
			break

	while True:
		if ',,' in txt_:
			txt_ = txt_.replace('--', '-')
		else:
			break

	while True:
		if '..' in txt_:
			txt_ = txt_.replace('//', '/')
		else:
			break

	if len(txt_.replace(',', '').replace(' ', '').strip()) > 0:
		if len(txt_) >= max_len:
			start = 0
			while True:
				if start >= len(txt_):
					break
				else:
					if len(txt_) >= start + max_len + 1:
						while True:
							if max_len>=50:
								if txt_[start+max_len] ==' ' or txt_[start+max_len] =='?' or txt_[start+max_len] ==',' or txt_[start+max_len] =='.' or txt_[start+max_len] =='!':
									sub_txt = txt_[start:start + max_len]
									if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
										if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
											sub_txt = sub_txt + '.'
										txt_list.append(sub_txt.strip())

									start += max_len
									max_len=60
									break
								else:
									max_len = max_len - 1
							else:
								sub_txt = txt_[start:start + max_len]
								if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
									if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
										sub_txt = sub_txt + '.'
									txt_list.append(sub_txt.strip())

								start += max_len
								max_len = 60
								break
					else:
						sub_txt = txt_[start:start + max_len]
						start += max_len

						if len(sub_txt.translate(str.maketrans('', '', string.punctuation))) > 0:
							if not (sub_txt.endswith('.') or sub_txt.endswith('?') or sub_txt.endswith('!')):
								sub_txt = sub_txt + '.'
							txt_list.append(sub_txt.strip())
		else:
			if not (txt_.endswith('.') or txt_.endswith('?') or txt_.endswith('!')):
				txt_ = txt_ + '.'
			txt_list.append(txt_.strip())
	print('Cleaning Text time: {}'.format(time.time() - start_time))
	return txt_list

def load_tacotron(hparams):
	model = Tacotron2(hparams).cuda()
	if hparams.fp16_run:
		model.decoder.attention_layer.score_mask_value = finfo('float16').min
	if hparams.distributed_run:
		model = apply_gradient_allreduce(model)
	return model



def plot_data(data, figsize=(16, 4)):
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1, len(data), figsize=figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect='auto', origin='bottom',
					   interpolation='none')
	return plt

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, sr, wav.astype(np.int16))


def load_models(tacotron_checkpoint_path, waveglow_checkpoint_path, hparams):
	tacotron = load_tacotron(hparams)
	tacotron.load_state_dict(torch.load(tacotron_checkpoint_path)['state_dict'])
	waveglow = torch.load(waveglow_checkpoint_path)['model']
	if hparams.fp16_run:
		waveglow.cuda().eval().half()
		tacotron.cuda().eval().half()
	for k in waveglow.convinv:
		k.float()
	denoiser = Denoiser(waveglow)
	return tacotron, waveglow, denoiser


def synthesize(tacotron_model, waveglow_model, denoiser, text_sequences, speaker_ids, language_ids, plot_mel=False, save_audio=False):
	start = time.time()
	mel_outputs, mel_outputs_postnet, _, alignments = tacotron_model.inference(text_sequences, speaker_ids=speaker_ids, language_ids=language_ids)


	print('Tacotron synthesize time: {}'.format(time.time() - start))
	start = time.time()
	with torch.no_grad():
		audio = waveglow_model.infer(mel_outputs_postnet, sigma=0.6606)

	print('Wavenet synthesize time: {}'.format(time.time() - start))
	audio = denoiser(audio, strength=0.06)[:, 0]

	if plot_mel:
		import matplotlib.pylab as plt
		plt.figure()
		plt.imshow(mel_outputs_postnet.float()[0].data.cpu().numpy())
		plt.savefig('inferencing_outputs.png')
		plt.close('all')
	if save_audio:
		start = time.time()
		save_wav(audio[0].data.cpu().numpy(), 'output_{}_{}.wav'.format(taco, wave), sr=hparams.sampling_rate)
		print('Audio --> .wav file saving time: {}'.format(time.time() - start))
	return audio

def get_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--text', type=str, default='안녀하세요. 저는 Russia 에서 왔습니다. 록스의 사투리 intelligent synthesis service입니다.',help='text to synthesize')
	parser.add_argument('-l', '--language', type=int, default=0,help='id of target language')
	parser.add_argument('-s', '--speaker', type=int, default=0,help='id of target speaker')
	parser.add_argument('-c', '--checkpoint', type=str, default='outdir/checkpoint_9608', required=True, help='path to checkpoint')
	args = parser.parse_args()
	return args

if __name__ =='__main__':
	args=get_arguments()
	hparams = create_hparams()
	hparams.distributed_run=False
	tacotron_checkpoint_path = args.checkpoint
	# waveglow_checkpoint_path = 'waveglow/checkpoints/waveglow_jeju_146000'
	waveglow_checkpoint_path = r'outdir/en-jeonla/waveglow_245000'
	### take checkpoint number to create audio file name
	taco = tacotron_checkpoint_path.split('_')[-1]
	wave = waveglow_checkpoint_path.split('_')[-1]

	tacotron, waveglow, denoiser = load_models(tacotron_checkpoint_path=tacotron_checkpoint_path, waveglow_checkpoint_path=waveglow_checkpoint_path, hparams=hparams)

	text = args.text
	lang_id=args.language
	speaker_id=args.speaker

	# text = 'If you depend on functionality not listed there, 제발 file an issue.'

	sequence, mask = text_to_sequence(text, cleaner=['english_cleaners'], lang=0)
	print(sequence)
	sequence = np.array(sequence).reshape(1,-1)
	sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
	start = time.time()


	speaker_ids = torch.LongTensor([speaker_id]) #### as we define (when training) 0 is korean speaker, 1 is english speaker
	speaker_ids = speaker_ids.unsqueeze(-1).expand((sequence.shape[0], sequence.shape[1]))
	language_ids = torch.LongTensor([lang_id]) #### 0 is korean, 1 is english
	language_ids = language_ids.unsqueeze(-1).expand((sequence.shape[0], sequence.shape[1]))
	# print(f'language_ids 1 {language_ids}')
	audio = synthesize(tacotron_model=tacotron, waveglow_model=waveglow, denoiser=denoiser, text_sequences=sequence, speaker_ids=speaker_ids, language_ids=language_ids, plot_mel=False, save_audio=True)
