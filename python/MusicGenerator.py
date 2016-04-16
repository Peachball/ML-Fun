
from WikiPageGenerator import LSTM
import platform
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

def downloadVideo(indexs, videos=getYoutubeVideoUrls()):
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
	if platform.system() == 'Linux':
		sp.call(['avconv', '-i', filename, '-f', 'wav', outputFilename])
	else:
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

def checkCompletion():
	import os.path
	missing = []
	for i in range(200):
		if not os.path.isfile('musicDataSet/' + str(i) + '.mp3'):
			print("Uh oh: " + str(i))
			missing.append(i)
	return missing

def repair():
	miss = checkCompletion()
	downloadVideo(miss)

INPUTSIZE = 100

def convertFileToDataSet(index, inputsize=100):
	filename = 'convertedDataSet/' + str(index) + '.wav'
	samplerate, data = wavUtil.read(filename)
	dataset = []
	#Get rid of the early silence parts
	index1 = 0
	index2 = -1
	while data[index1][0] == 0 and data[index1][1] == 0:
		index1 += 1
	while data[index2][0] == 0 and data[index2][1] == 0:
		index2 -= 1
	data = data[index1:(index2+1)]
	data = data.flatten()
	data = np.array(data)
	data.resize(data.shape[0]//100, inputsize)
	x = data
	y = data[100:]
	return x, y

def convertFilesToDataSet(indexs, inputsize=100, dataset=None):
	if dataset==None:
		ds = SequentialDataSet(inputsize, inputsize)
	for i in indexs:
		convertFileToDataSet(i, dataset=ds)
	return dataset

def generateMusicFile(net, name="test.wav", length=10000, noise=1e-9):
	output = []
	output.append((np.random.rand(1, 100) - 0.5) * noise)
	for i in range(length/100):
		output.append(net.predict(output[-1] + (np.random.rand(1, 100) - 0.5) * noise))
	

if __name__ == '__main__':
	x, y = convertFileToDataSet(0, inputsize=1000)
	lstm = LSTM(1000, 1000, 1000, out_type='linear', rprop=True, verbose=True, alpha=0.01, momentum=0.5,
			init_size=10, cell_size=[None])
	
	subset = 1
	s_error = 100
	sizeofset = 1000#y.shape[0]
	while True:
		s_error = 1e9
		while s_error > 1000:
			s_error = lstm.learn(x[:sizeofset], y[:sizeofset])
			print(s_error, sizeofset)

