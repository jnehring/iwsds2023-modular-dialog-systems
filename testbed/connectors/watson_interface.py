# connector create a chatbot in ibm watson assistant and chat with it using the web api
# author: jan nehring, 12/2020

import json
from ibm_watson import AssistantV1, AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import logging
from testbed.connectors.connector import ChatbotConnector
from testbed.connectors.chatbot_response import IntentClassifierResponse
import traceback
import time

config = {
    "watson_workspace_id":"ca3a9786-5d97-42e9-91f7-998ca0e825d5",
    "watson_api_key": "CTJBERpBGdpqdXdwJSJYONh2KlMBWNsWFM6HwwXq1UOs",
    "watson_service_url": "https://api.eu-gb.assistant.watson.cloud.ibm.com/"
}


class WatsonAPIInterface(ChatbotConnector):
    def __init__(self, api_key, service_url, workspace_id, name="watson"):
        self.name=name
        self.service_url=service_url
        self.workspace_id=workspace_id
        self.api_key=api_key

        self.authenticator = IAMAuthenticator(self.api_key)
        self.assistant_v1 = AssistantV1(
            version='2020-04-01',
            authenticator = self.authenticator
        )
        self.assistant_v1.set_service_url(self.service_url)


    # create watson interface with parameters from config
    @staticmethod
    def createInstance():
        try:
            return WatsonAPIInterface(config["watson_api_key"], config["watson_service_url"], config["watson_workspace_id"])
        except:
            traceback.print_exc()
            logging.info("Couldn't connect with IBM_WATSON!!")
            return None


    # delete any intent of your choice
    def deleteIntents(self):
        logging.info("list intents for deletion")

        response=self.assistant_v1.list_intents(
            workspace_id=self.workspace_id
        ).get_result()
        logging.info("retrieved {} intents".format(len(response["intents"])))

        for intent in response["intents"]:
            intent_name=intent["intent"]
            logging.info("delete intent " + intent_name)
            response=self.assistant_v1.delete_intent(
                workspace_id=self.workspace_id,
                intent=intent_name
            ).get_result()


    # creates ibtebt of your choice
    def createIntent(self, intent, examples):
        logging.info("create intent " + intent)
        response=self.assistant_v1.create_intent(
            workspace_id=self.workspace_id,
            intent=intent,
            examples=examples
        ).get_result()


    # list all dialog nodes
    def listDialogNodes(self):
        return self.assistant_v1.list_dialog_nodes(
            workspace_id=self.workspace_id
        ).get_result()


    # delete all dialog nodes
    def deleteDialogNodes(self):
        nodes=self.listDialogNodes()
        logging.info("retrieved {} existing dialog nodes".format(len(nodes["dialog_nodes"])))
        for node in nodes["dialog_nodes"]:
            logging.info("delete dialog node")
            response=self.assistant_v1.delete_dialog_node(
                workspace_id=self.workspace_id,
                dialog_node=node["dialog_node"]
            ).get_result()


    # write all intents from the dataframe to watson assistant
    def createIntentsFromDataframe(self, df):
        intents=df["new_intent"].unique()
        for intent in intents:
            examples=[]
            used_utterances=set()
            for ix, row in df[df["new_intent"]==intent].iterrows():
                if row["utterance"] in used_utterances:
                    continue

                used_utterances.add(row["utterance"])
                examples.append({
                    "text": row["utterance"]
                })
            self.createIntent(intent, examples)


    # create one dialog node per intent
    def createDialogNodesFromDataframe(self, df):
        intents=df["new_intent"].unique()
        for intent in intents:
            logging.info("create dialog node for intent " + intent)
            response=self.assistant_v1.create_dialog_node(
                workspace_id=self.workspace_id,
                dialog_node=intent,
                conditions='#'+intent,
                title=intent
            ).get_result()


    # startup method
    def setup_data(self, df):
        self.deleteIntents()
        self.deleteDialogNodes()
        self.createIntentsFromDataframe(df) 
        self.createDialogNodesFromDataframe(df)


    # send API requests for each utterance
    def chat(self, message):
        response = self.assistant_v1.message(
            workspace_id=self.workspace_id,
            input={
                'text': message
            }
        ).get_result()
        if len(response["intents"]) == 0:
            count = 0
            while count <1 and len(response["intents"]) == 0:
                response = self.assistant_v1.message(
                    workspace_id=self.workspace_id,
                    input={
                        'text': message
                    }
                ).get_result()
                count += 1
                time.sleep(1)

            if len(response["intents"]) == 0:
                response = IntentClassifierResponse(self.name, "", 0.0)
                return "", 0.0
            else:
                intent=response["intents"][0]
                response = IntentClassifierResponse(self.name, intent["intent"], intent["confidence"])
                return intent["intent"], intent["confidence"]
        else:
            intent=response["intents"][0]
            response = IntentClassifierResponse(self.name, intent["intent"], intent["confidence"])
            return intent["intent"], intent["confidence"]


    