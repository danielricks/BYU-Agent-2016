
import sys, pickle
import numpy as np
import random as rand

class verbFinder:

	def __init__(self):
		self.verbs = {}
		self.nouns = {}
		self.preps = {}
		self.unique_mod = ''

	def saveNounsToFile(self):
		pickle.dump(self.nouns, open(self.unique_mod + '_nouns.p', 'wb'))

	def addNounsFromFile(self, otherNounPickleFilename):
		otherNouns = pickle.load(open(otherNounPickleFilename, 'rb'))
		for noun in otherNouns:
			if noun in self.nouns:
				for verb in otherNouns[noun]:
					if verb in self.nouns[noun]:
						self.nouns[noun][verb] += otherNouns[noun][verb]
					else:
						self.inc_noun(noun, verb)
			else:
				for verb in otherNouns[noun]:
					self.nouns[noun] = {}
					if verb not in self.nouns[noun]:
						self.nouns[noun][verb] = 0
					self.nouns[noun][verb] += otherNouns[noun][verb]

	def savePrepsToFile(self):
		pickle.dump(self.preps, open(self.unique_mod + '_preps.p', 'wb'))

	def addPrepsFromFile(self, otherPrepPickleFilename):
		otherPreps = pickle.load(open(otherPrepPickleFilename, 'rb'))
		for Prep in otherPreps:
			if prep in self.preps:
				for verb in otherPreps[prep]:
					if verb in self.preps[prep]:
						self.prep[prep][verb] += otherPreps[prep][verb]
					else:
						self.inc_prep(prep, verb)
			else:
				for verb in otherPreps[prep]:
					self.preps[prep] = {}
					if verb not in self.preps[prep]:
						self.preps[prep][verb] = 0
					self.preps[prep][verb] += otherPreps[prep][verb]

	def saveVerbsToFile(self):
		pickle.dump(self.verbs, open(self.unique_mod + '_verbs.p', 'wb'))

	def addVerbsFromFile(self, otherVerbPickleFilename):
		otherVerbs = pickle.load(open(otherVerbPickleFilename, 'rb'))
		for verb in otherVerbs:
			if verb in self.verbs:
				for word in otherVerbs[verb]:
					if word in self.verbs[verb]:
						self.verbs[verb][word] += otherVerbs[verb][word]
					else:
						self.inc_verbs(verb, word)
			else:
				for word in otherVerbs[verb]:
					self.verbs[verb] = {}
					if word not in self.verbs[verb]:
						self.verbs[verb][word] = 0
					self.verbs[verb][word] += otherVerbs[verb][word]

	def verbsForNoun(self, noun):
		return self.nouns[noun]

	def verbsForPrep(self, prep):
		return self.preps[prep]

	def wordsForVerb(self, verb):
	        if verb in self.verbs.keys():
		     return self.verbs[verb]
		return {}

	def numDependencies(self, verb, word):
		return self.verbs[verb][word]

	def inc_verb(self, verb, word):
		if verb not in self.verbs:
			self.verbs[verb] = {}
		if word not in self.verbs[verb]:
			self.verbs[verb][word] = 0
		self.verbs[verb][word] += 1

	def inc_noun(self, noun, verb):
		if noun not in self.nouns:
			self.nouns[noun] = {}
		if verb not in self.nouns[noun]:
			self.nouns[noun][verb] = 0
		self.nouns[noun][verb] += 1

	def inc_prep(self, prep, verb):
		if prep not in self.preps:
			self.preps[prep] = {}
		if verb not in self.preps[prep]:
			self.preps[prep][verb] = 0
		self.preps[prep][verb] += 1

	# Parse a pre-processed set of lines
	def parseFile(self, inputFile):
		f = open(inputFile, 'r')
		sentence = []
		for line in f:
			if len(line) > 10:
				word = {}
				textLine = line.split()
				word['index'] = textLine[0]
				word['text'] = textLine[1]
				word['type'] = textLine[3]
				word['parent'] = textLine[6]
				if word['type'] == 'NOUN' or word['type'] == 'VERB' or word['type'] == 'ADP':
					sentence.append(word)
			else:
				if len(sentence) > 0:
					for w1 in sentence:
						#find all words with a verb as the parent
						if w1['type'] == 'VERB':
							for w2 in sentence:
								if w2['parent'] == w1['index'] and w2['type'] != 'VERB':
									self.inc_verb(w1['text'], w2['text'])
									if w2['type'] == 'NOUN':
										self.inc_noun(w2['text'], w1['text'])
									if w2['type'] == 'ADP':
										self.inc_prep(w2['text'], w1['text'],)
				sentence = [] #beginning of new sentence
		f.close()


