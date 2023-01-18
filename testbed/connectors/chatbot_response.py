# response types of the different connectors
#
# author: jan nehring, 01/2021

import ast
import jsonpickle
import ast
from typing import List

class ChatbotResponse:
    
    def __init__(self):
        self.agent_name=None
        
    def __repr__(self):
        return type(self).__name__ + " " + str(self.__dict__)

    def serialize(self) -> str:
        return jsonpickle.encode(self)

    @staticmethod
    def unserialize(json_str : str):
        return jsonpickle.decode(json_str)

    @staticmethod
    def unserialize_list(list_str : str):
        list_jsonp = ast.literal_eval(list_str)
        l=[]
        for jsonp_str in list_jsonp:
            l.append(ChatbotResponse.unserialize(jsonp_str))
        return l

class IntentClassifierResponse(ChatbotResponse):

    def __init__(self):
        pass

    def __init__(self, agent_name, intent, confidence):
        self.agent_name=agent_name
        self.intent=intent
        self.confidence=confidence
