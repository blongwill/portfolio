
from __future__ import print_function
import random
import re


##### Labda function for Alexa Framework
###### Uses python 3.6
###### Benny Longwill LING 575
###### Implement an FST-based System-Initiative Dialog
###### Spanish Quiz -- Tests Spanish Prepositions 'Por' and 'Para'


# --------------- Helpers that build all of the responses ----------------------

def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'SSML',
            'ssml': "<speak>"+ output +" </speak>" 
        },
        'card': {
            'type': 'Simple',
            'title': "SessionSpeechlet - " + title,
            'content': "SessionSpeechlet - " + output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'SSML',
                'ssml': "<speak>"+ reprompt_text +" </speak>" 
            }
        },
        'shouldEndSession': should_end_session
    }

def build_response(session_attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }


# --------------- Functions that control the skill's behavior ------------------

######## Function that allows funcitonality for changing question category during Quiz
def get_category_response(intent, session):
    
    card_title = "category"
    
    session_attributes = session.get('attributes')
   
    #### Gets the next category name
    new_category_name=intent.get('slots').get('category_response').get('value').lower()
    session_attributes['current_category']=new_category_name
    
    ######## Checks for previous question prompt, only changes after current question 
    if re.search('Question \d:', session['attributes']['previous_prompt']):
        speech_output = "Okay, we will switch to " + new_category_name + " next, but first let's finish the current question. " + session_attributes['previous_prompt']
    else:
        speech_output = get_question(session_attributes)[0]
        session_attributes['previous_prompt']=speech_output
 
  
    reprompt_text = "Sorry, I didn't get that, " + speech_output
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

####### Generates question from dictionary, if no more available in category then 'dummy' question created to ask for different category
def get_question(session_attributes):
    current_category=session_attributes['current_category']
    if len(session_attributes['questions'][current_category]) > 0:
        session_attributes['previous_question']=session_attributes['current_question']
        session_attributes['current_question']=session_attributes['questions'][current_category].pop(0)
        session_attributes['questions_seen']+=1
    else:
        session_attributes['current_question'] = ("There are no more questions in that category, please choose another category", "Por")
    return session_attributes['current_question']

### Handles 'por' or 'para' responses and awards/subtracts points accordingly. Also takes bets into account
def get_answer_response(intent, session):
    
    card_title = "answer"
    
    session_attributes = session.get('attributes')
    
    #### Gets the next category name
    user_answer=intent.get('slots').get('answer_response').get('value').lower()
    true_answer=session_attributes['current_question'][1]
    current_bet=session_attributes['current_bet']
    
    #### Checks model question answer as part of question tuple
    if user_answer==true_answer:
        speech_output = "Correct! "
        session_attributes['score']+=1
        session_attributes['points']+=session_attributes['current_question'][2]
        if current_bet>0: session_attributes['points']+=current_bet
        
    else:
        speech_output = "That is incorrect. "
        session_attributes['points']-=session_attributes['current_question'][2]
        if current_bet>0: session_attributes['points']-=current_bet
    
    session_attributes['current_bet']=0
    print("Fjdlskfjdklsjflkdsjfldjskfldjslfdfdsf")
    print(len([i for i in session_attributes['questions'].values() if len(i) > 0]))
    
    ######## Needs to be fixed to work with any number of questions
    #if (len(session_attributes['questions']['what would you use'])<1 and len(session_attributes['questions']['uses'])<1 and len(session_attributes['questions']['fill in the blank'])<1) or session_attributes['points']<0:   
    if len([i for i in session_attributes['questions'].values() if len(i) > 0])<1: 
        speech_output="Congratulations you completed the quiz! " + endGame(session)
        reprompt_text=""
        should_end_session=True 
    elif session_attributes['points']<0:
        
        speech_output="I'm sorry you are unable to complete the quiz because your points dropped to zero. " + endGame(session)
        reprompt_text=""
        
        should_end_session=True 
    else:   
        #Update current question    
        session_attributes['current_question']=get_question(session_attributes)
        #Update current propmpt 
        session_attributes['previous_prompt']=session_attributes['current_question'][0]
        
        reprompt_text = "Sorry, I didn't get that, " + session_attributes['previous_prompt']
        speech_output += session_attributes['previous_prompt']
        should_end_session=False
        
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

##### Creates end game text
##### If User gives name then has the option of updating info to S3 or boto
##### Outputs Thank you message and score with accuarcy percentage
def endGame(session):
    session_attributes=session['attributes']
    score = session_attributes['score']
    points = session_attributes['points']
    questions_seen = session_attributes['questions_seen']
    
    percentage=str(round((score/questions_seen)*100, 2))
    
    
    ######## For recording user statistics in 'Hall-Of_Fame' database
    ### Alexa only allows writing to files in tmp which does not persist between sessions.
    ##As a result Writing to file requires setting up a bucket for S3 and must be set up through Amazon.
    
    '''
    if 'user_name' in session_attributes:
        hall_of_fame_file = open("/tmp/hall_of_fame.txt","w+")
        hall_of_fame_lines = hall_of_fame_file.readlines()
        print(str(hall_of_fame_file.readlines()))
        
        hall_of_fame_lines.append(session_attributes['user_name'] + " " + str(score) + "/" + str(questions_seen) + " " + percentage + "% " + str(points))
        split_lines=[line.split() for line in hall_of_fame_lines]
        print(split_lines)
        sorted(split_lines, key=lambda x: x[-1])
        hall_of_fame_file.write( "fjdklsjfkldsjfl" )
        print(str(hall_of_fame_file.readlines()))
        hall_of_fame_file.close()
    '''    
    plural_questions_tag=""
    if int(questions_seen) > 1 : 
        plural_questions_tag="s"
    
        
    
    
    return "Thanks for playing the Spanish Quiz! Your final score is " + str(score) + ". You were " + str(percentage) + "% accurate on " + str(questions_seen) + " question"+ plural_questions_tag + ". You had " + str(points) + " points" 

##### Users have the option to place a bet at any time for the immediatly followin question
##### Method validates their bet amount as a digit. Users cannot bet more points than they have
##### Best only last for the following question and then are reset to zero
def get_bet_response(intent,session):
    
    card_title = "bet"
    
    session_attributes = session.get('attributes')
    
    bet_amount_str=intent.get('slots').get('bet_amount').get('value').lower()
    
    if bet_amount_str.isdigit():
        bet_amount_int = int(bet_amount_str)
        if bet_amount_int <= session_attributes['points'] and bet_amount_int>0:
            speech_output="Excellent, you have bet " + bet_amount_str + " on the next question."
            session_attributes['current_bet']=bet_amount_int
        else:
            speech_output="You do not have sufficient points to bet that much."
        
    else:
        print("That's not an int!")
        speech_output="I'm sorry, how much did you want to bet?"
        
    
    reprompt_text = session_attributes['previous_prompt']
    
    should_end_session=False
   
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

### Users have the option to ask how many points they have accumulated during any points during the quiz
### Alexa outputs the number of points
def get_points_response(session):
    
    card_title = "points"
    
    session_attributes = session.get('attributes')
    
    speech_output = "You have a total of " + str(session_attributes['points']) + " points" 
    
    reprompt_text = session_attributes['previous_prompt']
    
    should_end_session=False
   
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

###### Alexa gives an eplanation of a specific category if identified.
###### If no specific category is given, then Alexa just tells which categories are remaining
def get_category_information_response(intent,session):
    
    card_title = "categories"
    
    session_attributes = session.get('attributes')
    
    if 'value' in intent.get('slots').get('category_response'):
        category=intent.get('slots').get('category_response').get('value').lower()
        
        speech_output = session_attributes['category_explanations'].get(category)
        print("it is this " + session_attributes['category_explanations'].get(category))
    else:
        remaining_categories=[]
        for key, value in session_attributes['questions'].items():
            if len(value)>0:
                remaining_categories.append(key)
       ######## When creatin ga list of more than 1 item, then an "and" is included 
        if len(remaining_categories) >1:        
            remaining_categories.insert(-1, "and")
            
        speech_output = "You still have: " + ", ".join(remaining_categories) 
    
    reprompt_text = session_attributes['previous_prompt']
    
    should_end_session=False
   
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

##### if the user says por or para outside of the context of a question then a prompt is given to wait until a question is asked   
def get_oot_answer_response(intent, session):
    
    card_title = "oot answer"
    
    session_attributes = session.get('attributes')
   
    #### Gets the next category name
    speech_output="Please wait for the question before responding..." + session_attributes['previous_prompt']
 
  
    reprompt_text = "Sorry, I didn't get that, " + speech_output
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

##### Response for when the user gives a name (Name is used to record player stats in hall of fame)
def get_inform_name_response(intent, session):
    
    card_title = "name"
    
    session_attributes = session.get('attributes')
   
    user_name=intent.get('slots').get('user_name').get('value').upper()
   
    session_attributes['user_name'] = user_name
    
    #### Gets the next category name
    speech_output="Okay, I'll be sure to add you to the Hall of Fame!"
 
    reprompt_text = session_attributes['previous_prompt']
    
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))          

##### For when a player requests Alexa repeat, she will repeat most previous recorded prompt. Not necessarily the most recent thing said.
def get_repeat_response(intent, session):
    
    card_title = "repeat"
    
    session_attributes = session.get('attributes')
   
    speech_output= session_attributes['previous_prompt']
 
  
    reprompt_text = speech_output
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))   

##########When a user as for an explanation about the game or prepositions used
def get_preposition_response(intent, session):
    
    card_title = "preposition explanation"
    
    session_attributes = session.get('attributes')
   
    speech_output= session_attributes['preposition_explanation']
 
  
    reprompt_text = session_attributes['previous_prompt']
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
        
########## When a user asks why that is the answer alexa explains or gives an example correspoinding to the previous question
def get_question_explanation_response(intent, session):
    
    card_title = "question explanation"
    
    session_attributes = session.get('attributes')
    
    if 'previous_question' in session_attributes:
        speech_output= session_attributes['previous_question'][3]
    else:
        speech_output= "I didn't give you a quesiton?"
    
  
    reprompt_text = session_attributes['previous_prompt']
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))   

##### Starting prompt, initializes starting prompt
def get_welcome_response(session):
    """ If we wanted to initialize the session to have some attributes we could
    add those here
    """
    session_attributes = session.get('attributes')
    card_title = "Welcome"
    
    category_names=", ".join(session_attributes['questions'].keys()) ### Allows for more general framework
   
    
    session_attributes['previous_prompt']="Our categories are: "+ category_names + ". Where would you like to begin?"
    
    speech_output = session_attributes['welcome_prompt'] +  session_attributes['previous_prompt']
    
    # If the user either does not reply to the welcome message or says something
    # that is not understood, they will be prompted again with this text.
    reprompt_text = "I don't know if you heard me, Which category would you like?"
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

###### Only really used when a user stops the game before finishing
def handle_session_end_request():
    card_title = "Session Ended"
    speech_output = "Sorry to see you go! Come back soon!"
 
    # Setting this to true ends the session and exits the skill.
    should_end_session = True
    return build_response({}, build_speechlet_response(
        card_title, speech_output, speech_output, should_end_session))

# --------------- Events ------------------

def on_session_started(session_started_request, session):
    """ Called when the session starts.
        One possible use of this function is to initialize specific 
        variables from a previous state stored in an external database
    """
    
    ##### Tag for the beep noise during the fill in the blank statements
    beep_tag="<say-as interpret-as='expletive'> p </say-as>"
    
    ####### SSML tags for creating the spanish voices and accents in the output speech
    spanish_voice_tags=["<voice name='Conchita'>","<voice name='Enrique'>" ]
    spanish_accent_tag="<lang xml:lang='es-ES'>"
    end_voice_tag="</voice>"
    end_accent_tag="</lang>"
    por= "<voice name='Conchita'><lang xml:lang='es-ES'>Por</lang></voice>"
    para="<voice name='Conchita'><lang xml:lang='es-ES'>Para</lang></voice>"
   
   
    #### Session attributes
    session['attributes']={'previous_prompt': "" }
    session['attributes'].update({'current_category': "" })
    session['attributes'].update({'current_question': () })
    session['attributes'].update({'previous_question': () }) #### Used mainly for explanation of previous question clarification
    session['attributes'].update({'points': 300 }) ### Starting points 
    session['attributes'].update({'score': 0 }) ## Score is striclty count of correct answers
    session['attributes'].update({'current_bet': 0 }) #### Amount user bets is reset aftear each resolved question
    session['attributes'].update({'questions_seen': 0 })
    
    ########################## Text to speech for speech  prompt output
    session['attributes'].update({ 'preposition_explanation' : "Learning the difference between " + por + " and " + para + "may be challenging. In Spanish, these \
     are two prepositions that can have many meanings in English, including: 'for', 'by', 'on', 'through', 'because of', 'in exchange for', 'in order to'. Fortunately with this Quiz-Game you can easily master the differences!" })
    
    session['attributes'].update({'welcome_prompt' : "Welcome to the Spanish Quiz! Please select a category and respond to the questions with the appropriate preposition: " + por + " or " + para + " to obtain points. You shall begin with "+  str(session['attributes']['points']) +" points, so if you're feeling confident,\
        place a bet on any question to gain points more quickly. Beware! If you answer incorrectly you may lose points and if they drop to zero the quiz will end! Also, feel free to ask about the prepositions, categories, or to change the category. "})
   
    session['attributes'].update({'category_explanations' : {
        'what would you use': "A sentence scenario will be given in English. Please provide the Spanish word "+ por + " or "+ para +"  that corresponds to the preposition. " ,
        'fill in the blank' : "A sentence will be given in Spanish and have a word replaced with "+beep_tag +" ...Please listen carefully and provide the missing Spanish preposition: either "+ por + " or "+ para ,
        'uses' : "You will be given a common usage of "+ por + " or "+ para + ". Please state the corresponding preposition" }})       
       
    
    ############## Questions are represented as tuples in a list: [0]=question prompt [1]=true answer [2]=point value of question [3]=answer clarification/explanation
    session['attributes'].update({'questions' : {
        'what would you use': [("'what-would-you-use' Question 1: Thanks for the gift you gave me" , "por", 100, por+" is used to express gratitude for something"),
        ("'what-would-you-use' Question 2: I cook at home in order to eat well"  , "para", 200, para +" is used to mean 'in order to' or 'for the purpose of' "), ("'what-would-you-use' Question 3: Is this package for me?" , "para", 300, para + " is used to indicate a recipient"),
        ("'what-would-you-use' Question 4: I went to the zoo through the park" , "por",400, por + " is used traversing 'through,' 'along,' 'by' or 'in the area of' a place "), ("'what-would-you-use' Question 5: We usually leave for school at 7:30"  , "por",500, por + " is used to express a length of time ") ],
    
        'fill in the blank' : [("'fill-in-the-blank' Question 1: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)]+ spanish_accent_tag + "Necesito salir " +beep_tag+ " la casa a las siete." + end_accent_tag+ end_voice_tag , "para", 100, para + " is used to indicate destination"), 
        ("'fill-in-the-blank' Question 2: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag +   "El editor me llamó "+beep_tag+" teléfono." +end_accent_tag +end_voice_tag , "por", 200, por + " is used for means of communication or transportation"),
        ("'fill-in-the-blank' Question 3: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag +  beep_tag + " eso, los estudiantes leen el libro." + end_accent_tag + end_voice_tag , "por",300, por + " is used in that idiomatic expression to mean therefore"),
        ("'fill-in-the-blank' Question 4: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag +  beep_tag + " ir a la escuela, tienes que tomar el tren."+ end_accent_tag + end_voice_tag , "para",400, para + " is used to mean 'in order to' or 'for the purpose of'"), 
        ("'fill-in-the-blank' Question 5: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "España es conocida "+beep_tag +" su historia tan interesante."+ end_accent_tag + end_voice_tag , "por",500, por + " is used to express 'due to' or the motivations or reasons for something")],

        
        'uses' : [("'uses' Question 1: means of communication; period of time", "por", 100 , "It's like in the example: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "'Tengo que trabajar por ocho horas hoy.'" + end_accent_tag + end_voice_tag + "... I have to work for eight hours today "),
        ("'uses' Question 2: recipient of an action or object" , "para", 200 , "It's like in the example: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "'No, Bruno. Estos chocolates no son para los perros.'" + end_accent_tag + end_voice_tag + "... No, Bruno. These chocolates aren't for dogs."),
        ("'uses' Question 3: movement through a place" , "por", 300 , "It's like in the example: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "'Viajé por Francia y España.'" + end_accent_tag + end_voice_tag + "... I travelled through France and Spain" ), 
        ("'uses' Question 4: purpose; opinion" , "para", 400 , "It's like in the example: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "'Para mí, el español es más romántico que el italiano.'" + end_accent_tag + end_voice_tag + "... For me, Spanish is more romantic than Italian."),
        ("'uses' Question 5: movement toward a place" , "para",500, "It's like in the example: " + spanish_voice_tags[random.randint(0,len(spanish_voice_tags)-1)] + spanish_accent_tag + "'Este cuadro es para un museo en Madrid.'" + end_accent_tag + end_voice_tag + "... This painting is for a museum in Madrid.")] } })
       
    pass

    

def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """
    return get_welcome_response(session)


def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    if intent_name == "inform_category":
        return get_category_response(intent, session)
    elif intent_name == "inform_answer":
        ######## Uses 'Question \d:' as marker for actual question prompts
        if re.search('Question \d:', session['attributes']['previous_prompt']):
            return get_answer_response(intent, session)
        else:
            return get_oot_answer_response(intent,session)
    elif intent_name == "request_points":
        return  get_points_response(session)
    elif intent_name == "place_bet":
        return get_bet_response(intent,session)
    elif intent_name == "request_category_explanation":
        return get_category_information_response(intent,session)
    elif intent_name == "request_repeat":
        return get_repeat_response(intent, session)
    elif intent_name == "inform_name":
        return get_inform_name_response(intent, session)
    elif intent_name == "request_preposition_explanation":
        return get_preposition_response(intent, session)
    elif intent_name == "request_question_explanation":
        return get_question_explanation_response(intent, session)
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.StopIntent":
        return handle_session_end_request()
    elif intent_name == "AMAZON.FallbackIntent":
        ## Hopefully in case unkonwon response given and it will just repeat the previous prompt
        return get_repeat_response(intent,session)
    else:
        raise ValueError("Invalid intent")


def on_session_ended(session_ended_request, session):
    """ Called when the user ends the session.
    Is not called when the skill returns should_end_session=true
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # add cleanup logic here


# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    print("Incoming request...")

    """
    Uncomment this if statement and populate with your skill's application ID to
    prevent someone else from configuring a skill that sends requests to this
    function.
    """
    # if (event['session']['application']['applicationId'] !=
    #         "amzn1.echo-sdk-ams.app.[unique-value-here]"):
    #     raise ValueError("Invalid Application ID")

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']},
                           event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])
    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])
    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])