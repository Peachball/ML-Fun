import numpy
import subprocess as sp
import threading
import urllib.request
import pafy
from pyquery import PyQuery

doc = PyQuery(urllib.request.urlopen('https://www.youtube.com/watch?v=LiHUTGcYYFc&list=PL1zHNSEzATWIMsc2mMObsyJAx66HrH-Tv').read())

videos = doc('#playlist-autoscroll-list > li > a')

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
		output = open('ey.txt', 'a')
		output.write('something happened once\n')
	return

def readAudioFile(index):
	sp.call(['ffmpeg', '-i', '/input/file.mp3', '/output/file.wav'])

readAudioFile(0)

#Begin machine learning part
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer


