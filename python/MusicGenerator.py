import numpy as np
import subprocess as sp
import threading
import urllib.request
import pafy
from pyquery import PyQuery

def getYoutubeVideoUrls():
	doc = PyQuery(urllib.request.urlopen('https://www.youtube.com/watch?v=LiHUTGcYYFc&list=PL1zHNSEzATWIMsc2mMObsyJAx66HrH-Tv').read())
	videos = doc('#playlist-autoscroll-list > li > a')
	return videos

def downloadVideo(indexs):
	for index in indexs:
		link = 'https://youtube.com' + videos[index].attrib['href']
		p = pafy.new(link)
		stream = p.getbest()
		stream.download('musicDataSet/' + str(index) + '.mp3')
	return

def downloadAllMusic():
	for i in range(20):
		t = threading.Thread(target=downloadVideo, args=(range(i * 10, i * 10 + 10),))
		t.dameon = False
		t.start()
	return

def convertAudioFile(index):
	filename = "musicDataSet/" + str(index) + ".mp3"
	outputFilename = "convertedDataSet/" + str(index) + ".wav"
	sp.call(['ffmpeg', '-loglevel', 'panic', '-i', filename,  outputFilename])

def convertAudioFiles(ra):
	for i in ra:
		convertAudioFile(i)
	return

def convertAllAudioFiles():
	for i in range(20):
		t = threading.Thread(target=convertAudioFiles, args=(range(i*10,
			i*10+10),))
		t.dameon = False
		t.start()

#Begin machine learning part
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.structure.modules import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
import scipy.io.wavfile as wavUtil
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

INPUTSIZE = 100

def convertFileToDataSet(index, inputsize=100, dataset=None):
	filename = 'convertedDataSet/' + str(index) + '.wav'
	samplerate, data = wavUtil.read(filename)
	if dataset==None:
		dataset = SequentialDataSet(inputsize, inputsize)
	else:
		dataset.newSequence()
	data = data.flatten()
	for i in range(int(len(data)/inputsize)):
		bdata = data[i:(i+inputsize)]
		dataset.addSample(data[i:(i+inputsize)],
				(data[(i+inputsize):(i+2*inputsize)]))
	return dataset

def convertFilesToDataSet(indexs, inputsize=100, dataset=None):
	if dataset==None:
		ds = SequentialDataSet(inputsize, inputsize)
	for i in indexs:
		convertFileToDataSet(i, dataset=ds)
	return dataset

def generateMusicFile(network, name="test.wav", length=60):
	pass

try:
	lstm = NetworkReader.readFrom('musicgennet.xml')
	print('Loaded existing network')
except:
	lstm = buildNetwork(INPUTSIZE, 10, INPUTSIZE, hiddenclass=LSTMLayer,
		outclass=LinearLayer)
	print('Generated new network')

trainer = BackpropTrainer(lstm)
for i in range(200):
	print('Training...')
	print('Trained one iteration with error:' +
			str(trainer.trainOnDataset(convertFileToDataSet(i))))
	NetworkWriter.writeToFile(lstm, 'musicgennet.xml')
