# base class for all chatbot connectors
#
# author: jan nehring, 12/2020

from abc import ABC, abstractmethod

class ChatbotConnector(ABC):

    name=None

    @abstractmethod
    def chat(self, message):
        pass