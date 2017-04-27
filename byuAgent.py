#uses top n verbs from Wikipedia as its verb list (some hand optimization included)
#grabs objects from game text and inventory
#looks for verbs that match objects using one of:
#	w2v analogies
#	wikipedia co-occurance counts
#	verbFinder dependency counts
#seeks a verb that:
#	(A) satisfies the above search criteria
#	(B) is in the agent's verb list
#	(C) has not been tried with that object before
#remembers which verb/object combos produced reward/state changes using one of
#	simple counts
#	Qvalues

import agentBaseClass
import numpy as np
import time
import random as rand
import scholar.scholar as sch
import nltk
import re
import verbFinder

class byuAgent(agentBaseClass.AgentBaseClass):


	def __init__(self, initial_epsilon=1, training_cycles=1000):
		print("Initializing agent")
		
		agentBaseClass.AgentBaseClass.__init__(self, initial_epsilon, training_cycles)
		
		self.evaluation_metric = "ANALOGY"
		#self.evaluation_metric = "WIKIPEDIA_COOCCURANCE"
		#self.evaluation_metric = "DEPENDENCIES"

		print("Verb evaluation strategy is " + self.evaluation_metric)
		
		self.manipulation_list = ['throw', 'spray', 'stab', 'slay', 'open', 'pierce', 'thrust', 'exorcise', 'place', 'jump', 'take', 'make', 'read', 'strangle', 'swallow', 'slide', 'wave', 'look', 'dig', 'pull', 'put', 'rub', 'fight', 'ask', 'score', 'apply', 'take', 'knock', 'block', 'kick', 'step', 'break', 'wind', 'blow', 'crack', 'drop', 'blast', 'leave', 'yell', 'skip', 'stare', 'hurl', 'hit', 'kill', 'glass', 'engrave', 'bottle', 'pour', 'feed', 'hatch', 'swim', 'spray', 'melt', 'cross', 'insert', 'lean', 'sit', 'move', 'fasten', 'play', 'drink', 'climb', 'walk', 'consume', 'kiss', 'startle', 'shout', 'close', 'cast', 'set', 'drive', 'lift', 'strike', 'startle', 'catch', 'board', 'speak', 'think', 'get', 'answer', 'tell', 'feel', 'get', 'turn', 'listen', 'read', 'watch', 'wash', 'purchase', 'do', 'sleep', 'fasten', 'drag', 'swing', 'empty', 'switch', 'slip', 'twist', 'shoot', 'slice', 'read', 'burn', 'hop', 'rub', 'ring', 'swipe', 'display', 'scrub', 'hug', 'operate', 'touch', 'sit', 'sweep', 'fix', 'walk', 'crack', 'skip']
		self.manipulation_list += ['wait', 'point', 'light', 'unlight', 'use', 'ignite', 'wear', 'remove', 'unlock', 'lock', 'examine', 'inventory', '']
		self.navigation_list = ['north', 'south', 'west', 'east', 'northwest', 'southwest', 'northeast', 'southeast', 'up', 'down', 'enter', 'exit', 'drop']
		self.verb_list = self.manipulation_list + self.navigation_list
	
		if 'save' in self.verb_list:
			self.verb_list.remove('save') #to prevent agent from trying to save the game...		
		if 'quit' in self.verb_list:
			self.verb_list.remove('quit') #to prevent agent from trying to quit the game...		
		if 'restart' in self.verb_list:
			self.verb_list.remove('restart') #to prevent agent from trying to restart the game...		

	        #mod by ben for the additional prepositions
                self.preposition_list = ['with', 'in', 'at', 'above', 'under']
                #Verb and Preposition Dictionary
                self.VPD = {}


		if self.evaluation_metric == "ANALOGY":
			print("loading word2vec data...")
			self.scholar = sch.Scholar()
		
		#if self.evaluation_metric == "WIKIPEDIA_COOCCURENCE":
		#	self.corpus_name = "corpora/Wikipedia_first_100000_lines.txt"
		#	#self.corpus_name = "corpora/classic_books.txt"
		#	self.totalCount = {}
		#	for v in self.verb_list:
		#		self.totalCount[v] = 0.0
		#	f = open(self.corpus_name)
		#	for line in f:
		#		for v in self.verb_list:				
		#			if v in line:
		#				self.totalCount[v] = self.totalCount[v] + 1
		#	f.close()

		#if self.evaluation_metric == "DEPENDENCIES":
		#	self.verbFinder = verbFinder.verbFinder()
		
		print("loading verbFinder data...")
		self.verbFinder = verbFinder.verbFinder()
		self.verbFinder.addVerbsFromFile("agents/master_verbs.p")
		for v in self.verb_list:
			words = self.verbFinder.wordsForVerb(v)
			preps = []
			if len(words.keys()) > 0:
				for w in list(words.keys()):
					if w in self.preposition_list:
						preps.append(w)
			else:
				preps = ''
	 	
			if len(preps) == 0:
				preps = self.preposition_list
			#print(v)
			self.updatePrepositionDictionary(v, preps)
			#self.updatePrepositionDictionary(v, self.preposition_list)


		self.num_states = 10000
		self.last_state = ''
		self.current_state = ''
		self.last_narrative = ''
		self.current_narrative = ''
		self.last_verb = ''
		self.last_object = ''
		self.last_action = 'look'
		self.inventory_list = []
		self.inventory_text = ""
		self.TWO_WORD_OBJECTS = True
		self.inventory_count = 0
		self.look_flag = 0
		self.get_flag = 0
		self.packrat_count = 0 #am I just getting all all the time? (because the game narrative is too variable)
		self.inventory_count = 0 #am I just checking inventory? (because the game narrative is too variable)
		self.game_steps = 0
		self.exploration_counts = {}
		self.visited_states = []
		self.visited_narratives = []
		self.verbs_for_noun = {}

		self.alreadyTried = {}
		self.success = {}


        #Verb Noun Prep Noun, return list       
        def getVNPN(self):
                sents = []

                for v in self.verb_list:
                        for n in self.object_list:
                                for p in self.prep_list:
                                        for n2 in self.object_list:
                                                sentence = "{} {} {} {}".format(v, n, p, n2)
                                                sents.append(sentence)

                #for each string, check against vector matrix and delete bad strings

                return sents

        #look in box
        #Verb Prep Noun, return list    
        def getVPN(self):
                sents = []

                for v in self.verb_list:
                        for p in self.prep_list:
                                for n in self.object_list:
                                                sentence = "{} {} {}".format(v, p, n)
                                                sents.append(sentence)

                #for each string, check against vector matrix and delete bad strings

                return sents


        #fill a dictionary with a <str,set> combo
        def updatePrepositionDictionary(self,verb,prepSet):
                self.VPD[verb] = prepSet
                pass


        #using the dictionary, return a list of commands
        def getCommands(self):

                sents = []
                #Verb
                for v in self.verb_list:
                        #Noun
                        for obj in self.object_list:
                                #Dictionary of prepositions according to verbs
                                for key in self.VPD.keys():
                                        #set or list of prepositions
                                        for prep in self.VPD[key]:
                                                #second Noun
                                                for obj2 in self.object_list:
                                                        sentence = "{} {} {} {}". format(v, obj, prep, obj2)
                                                        sents.append(sentence)

                return sents


        #using the dictionary, return a list of commands
        def getCommands(self,verbs,objects):

                if '' in verbs:
                        verbs.remove('')
                if '' in objects:
                        objects.remove('')

                sents = []
                #Verb
                #print (verbs)
                #print (objects)
                #print (self.VPD)
                for v in verbs:
                        #Noun
                        for obj in objects:
                                #Dictionary of prepositions according to verbs
                                for key in self.VPD.keys():
                                        #set or list of prepositions
                                        for prep in self.VPD[key]:
                                                #second Noun
                                                for obj2 in objects:
                                                        sentence = "{} {} {} {}". format(v, obj, prep, obj2)
                                                        sents.append(sentence)

                return sents


	def state_index(self, game_text):
		#states are represented as a simple hash of the game text 
		return abs(hash(game_text))%self.num_states

	def find_objects(self, narrative):
		#Assume an object is manipulatable if it appears as a noun in the game text
		tokens = nltk.word_tokenize(narrative)
		tags = nltk.pos_tag(tokens)
		nouns = [word for word,pos in tags if word.isalnum() and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

		if self.TWO_WORD_OBJECTS == True:
			tokens = nltk.word_tokenize(narrative)
			tags = nltk.pos_tag(tokens)
			for i in range(0, len(tags) - 1):
				if (tags[i][1] == "JJ") and (tags[i+1][1] in ["NN", "NNP", "NNS", "NNPS"]):
					nouns.append(tags[i][0] + " " + tags[i+1][0])

		return nouns

	def get_wikipedia_verbs(self, obj, n):
		#returns the n most-commonly-co-occurring verbs from the 
		#wikipedia corpus (give or take a few)

		count = {}
		for v in self.verb_list:
			count[v] = 0.0

		f = open(self.corpus_name)
		for line in f:
			if obj in line:
				for v in self.verb_list:
					if v in line:
						count[v] = count[v] + 1
		f.close()
		
		for v in self.verb_list:
			count[v] = count[v]/self.totalCount[v]

		max_verbs = []

		while len(max_verbs) < n:
			for v in max_verbs:
				count[v] = -1
			max_val = max(count.values())
			max_verbs = max_verbs + [k for k in count if count[k] == max_val]
		
		return max_verbs

	def getTryList(self, game_text, input_object):

		#some objects are composed of two words (usually an adjective and an object)
		#If that is the case, then consider only the second word out of the pair
		obj = input_object
		if len(input_object.split()) > 1:
			obj = input_object.split()[-1]

		obj = obj.lower()

		#identify a set of verbs that seems to 'match' the current object of interest.
		#(This is accomplished using one of three different methods, all of which
		#rely on the Wikipedia corpus for the extraction of common-sense knowledge
		#about the relationship of verbs to specific objects.)
		#if self.evaluation_metric == "DEPENDENCIES":
		#	matching_verbs = self.verbFinder.verbsForWord(obj, 30)
		if self.evaluation_metric == "ANALOGY":
			if obj in self.verbs_for_noun.keys():
				matching_verbs = self.verbs_for_noun[obj]
			else:
				matching_verbs = self.scholar.get_verbs(obj, 30)
				for i in range(len(matching_verbs)):
					matching_verbs[i] = matching_verbs[i][:-3]
				self.verbs_for_noun[obj] = matching_verbs
		elif self.evaluation_metric == "WIKIPEDIA_COOCCURANCE":
			matching_verbs = self.get_wikipedia_verbs(obj, 30)
		else:
			print("ERROR: No match for evaluation metric " + self.evaluation_metric)
			input("")		

		tryList = []

		#we first try to manipulate the objects extracted from the game text,
		#so we look for the intersection between our manipulation list and the
		#wikipedia verbs that match this object
		for v in matching_verbs:
			if v in self.manipulation_list:
				if self.alreadyTried[game_text][input_object][v] == 0:
					tryList.append(v)

		#certain verbs are so useful that we ALWAYS include them in the try list
		if 'open' not in tryList and 'open' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['open'] == 0:
			tryList.append('open')
		if 'get' not in tryList and 'get' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['get'] == 0:
			tryList.append('get')
		if 'put' not in tryList and 'put' in self.alreadyTried[game_text][input_object].keys() and self.alreadyTried[game_text][input_object]['put'] == 0:
			tryList.append('put')

		#if we've tried everything we can think of, then we proceed
		#to try either (A) Things that worked, or (B) navigate elsewhere
		if len(tryList) == 0:
			if obj not in self.success[game_text].keys():
				self.success[game_text][obj] = {}
				for v in self.verb_list:
					self.success[game_text][obj][v] = 0.0
			tryList = list(self.success[game_text][obj].keys()) + self.navigation_list

		#if nothing seems to be working, then navigate away
		if len(tryList) == 0:
			tryList = self.navigation_list

		return tryList

	def getVerb(self, game_text, input_object):
		#returns a verb that:
		# (A) satisfies the active search criterion
		# (B) is in the agent's verb_list
		# (C) has not already been tried in this state with this object	

		tryList = self.getTryList(game_text, input_object)	

		vrb = rand.choice(tryList)
		#vrb = tryList[0]

		return vrb

	def chooseAction(self, game_text):

		if self.packrat_count > 0:
			self.packrat_count -= 1 #we've done something besides just 'get all'	

		if self.inventory_count > 0:
			self.inventory_count -= 1 #we've done something besides just 'get all'	
		
		objects = self.find_objects(game_text) + ['']
		obj = rand.choice(objects)

		#if this is a new state or object, then
		#initialize the proper dictionary keys
		if game_text not in self.alreadyTried.keys():
			self.alreadyTried[game_text] = {}

		if obj not in self.alreadyTried[game_text].keys():
			self.alreadyTried[game_text][obj] = {}
			for v in self.verb_list:
				self.alreadyTried[game_text][obj][v] = 0

		if game_text not in self.success.keys():
			self.success[game_text] = {}

		if obj not in self.success[game_text].keys():
			self.success[game_text][obj] = {}
			for v in self.verb_list:
				self.success[game_text][obj][v] = 0

		#Check to see whether the last action was successful
		#If it was, then remember that this was a useful action
		#('Success' is defined as eliciting a state change.)
		if self.last_state != self.current_state:
			if self.last_state not in self.success.keys():
				self.success[self.last_state] = {}
			if self.last_object not in self.success[self.last_state].keys():
				self.success[self.last_state][self.last_object] = {}
				for v in self.verb_list:
					self.success[self.last_state][self.last_object][v] = 0
			self.success[self.last_state][self.last_object][self.last_verb] = 1
		
		#choose the next action
		r = rand.randint(0, 10)
		if r == 0 and obj != '':
			#get a verb/preposition/object combo
			commands = self.getCommands(self.getTryList(game_text, obj), self.find_objects(game_text) + self.inventory_list)
			action = rand.choice(commands)
			return action
		else:
			#get a verb/object combination
			vrb = self.getVerb(game_text, obj)		
			self.alreadyTried[game_text][obj][vrb] = 1
			self.last_verb = vrb
			self.last_object = obj
			return vrb + ' ' + obj
			

	def take_action(self, narrative, evaluation_flag=False):
			
		self.game_steps += 1

		#every 1000 steps, reset the alreadyTried list
		#(this helps the agent try new things and keeps it from
		#gettibg 'stuck in a rut'. It also helps compensate for
		#unobservable state changes.)
		if self.game_steps%1000 == 0:
			for state in self.alreadyTried.keys():
				for obj in self.alreadyTried[state].keys():
					for vrb in self.alreadyTried[state][obj].keys():
						self.alreadyTried[state][obj][vrb] = 0

		#process results of look and inventory commands
		if self.last_action == "inventory":
			self.inventory_list = self.find_objects(narrative)
			self.inventory_text = narrative
			self.get_flag = 0
		elif self.last_action == "look":
			self.last_state = self.current_state
			self.last_narrative = self.current_narrative
			self.current_narrative = re.sub(r'\d+', '', narrative)
			#state is the narrative plus inventory
			self.current_state = re.sub(r'\d+', '', narrative + self.inventory_text)
			#print(self.current_state)
			#input("pause")

		#execute 'look' command every other step.
		#(This helps to make the state space more observable)
		if self.look_flag == 1:
			self.look_flag = 0
			self.last_action = "look"
			return "look"
		else:
			self.look_flag = 1

		#check inventory whenever we execute a 'get' command.
		#(inventory results are included as part of the state space)
		if self.last_verb == 'get':
			self.get_flag = 1
		if self.get_flag > 0 and self.inventory_count < 5:
			self.last_action = "inventory"
			self.last_verb = "inventory"
			self.inventory_count += 1
			self.get_flag = 0
			return "inventory"
	
		#try 'get all' in each new state	
		if self.current_narrative not in self.visited_narratives and self.packrat_count < 10:
			self.visited_narratives.append(self.current_narrative)
			self.last_action = 'get all'
			self.last_verb = 'get'
			self.last_object = 'all'
			self.get_flag = 1
			self.packrat_count += 1
			if self.packrat_count > 5: 
				self.packrat_count = 100
		else:			
			#select an action
			self.last_action = self.chooseAction(self.current_state)
			print ("Action is " + self.last_action.strip()) 
		
		return self.last_action.strip()

