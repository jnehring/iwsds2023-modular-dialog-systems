# connector create a chatbot in google dialogflow assistant and chat with it using the web api
# author: akhyar ahmed, 01/2021

import os
import google.cloud.dialogflow_v2 as gdf
import traceback
import logging
from google.api_core.exceptions import InvalidArgument
from testbed.connectors.connector import ChatbotConnector
from testbed.connectors.chatbot_response import IntentClassifierResponse

config = {
    "DialogFlow_session_id": "me",
    "DialogFlow_language_code": "en",
    "DialogFlow_project_id": "first-project-298809",
    "DialogFlow_api_key": "assets/google_private_key.json"
}

config_mod_all = {
    "DialogFlow_session_id": "me1",
    "DialogFlow_language_code": "en",
    "DialogFlow_project_id": "mod-ir-all",
    "DialogFlow_api_key": "assets/google_private_key_mod_ir_all.json"
}

config_mod_self = {
    "DialogFlow_session_id": "me2",
    "DialogFlow_language_code": "en",
    "DialogFlow_project_id": "mod-ir-self",
    "DialogFlow_api_key": "assets/google_private_key_mod_ir_self.json"
}

class DialogFlowAPIInterface(ChatbotConnector):
    def __init__(self, api_key, project_id, language, session_id, name="dialogflow"):
        self.name = name
        self.api_key = api_key
        self.project_id = project_id
        self.language = language
        self.session_id = session_id

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.api_key

        self.session_client = gdf.SessionsClient()
        self.session = self.session_client.session_path(self.project_id, self.session_id)

    @staticmethod
    def createInstance():
        try:
            return DialogFlowAPIInterface(config["DialogFlow_api_key"], config["DialogFlow_project_id"],
                                      config["DialogFlow_language_code"], config["DialogFlow_session_id"])
        except:
            traceback.print_exc()
            logging.info("Couldn't connect with Google Dialogflow!!")
            return None


    def createInstance_all():
        try:
            return DialogFlowAPIInterface(config_mod_all["DialogFlow_api_key"], config_mod_all["DialogFlow_project_id"],
                                      config_mod_all["DialogFlow_language_code"], config_mod_all["DialogFlow_session_id"])
        except:
            traceback.print_exc()
            logging.info("Couldn't connect with Google Dialogflow!!")
            return None


    def createInstance_self():
        try:
            return DialogFlowAPIInterface(config_mod_self["DialogFlow_api_key"], config_mod_self["DialogFlow_project_id"],
                                      config_mod_self["DialogFlow_language_code"], config_mod_self["DialogFlow_session_id"])
        except:
            traceback.print_exc()
            logging.info("Couldn't connect with Google Dialogflow!!")
            return None


    # create any intent of your choice
    def createIntent(self, intents, all_examples):
        for intent in intents:
            client = gdf.services.intents.IntentsClient()
            parent = client.common_project_path(self.project_id) + "/agent"

            training_phrases = []
            intent_utterances = all_examples[all_examples["new_intent"] == intent]
            utterances = intent_utterances["utterance"]
            for utterance in utterances:
                part = [gdf.types.Intent.TrainingPhrase.Part(text=utterance)]
                training_phrase = gdf.types.Intent.TrainingPhrase(parts=part)
                training_phrases.append(training_phrase)

            parent_path = "projects/" + self.project_id + "/agent"
            
            intent_req = gdf.types.Intent(
                display_name=intent,
                training_phrases=training_phrases)
            
            create_intent_request = gdf.types.CreateIntentRequest(
                parent=parent,
                intent=intent_req,
                language_code=self.language)
            try:
                intent_response = client.create_intent(parent=parent, intent=intent_req)
            except InvalidArgument:
                raise
        return


    # create an agent
    def createAgent(self, name):
        client_agent = gdf.services.agents.AgentsClient()
        agent = gdf.types.Agent(
            parent="projects/" + self.project_id,
            display_name=name,
            default_language_code=self.language,
            time_zone="Europe/Madrid")

        set_agent_request = gdf.types.SetAgentRequest(agent=agent)
        try:
            agent_response = client_agent.set_agent(agent=agent)
        except InvalidArgument:
            raise
        return
    

    # delete any intent of your choice
    def deleteAgent(self):
        client_agent = gdf.services.agents.AgentsClient()
        delete_agent_request = gdf.types.DeleteAgentRequest(parent="projects/" + self.project_id)
        try:
            deleteAgent_response = client_agent.delete_agent(parent="projects/" + self.project_id)
        except InvalidArgument:
            raise
        return


    # deletes all default nlu-fallback intents
    def deleteDefaultIntents(self):
        client_intent = gdf.services.intents.IntentsClient()
        path = "projects/" + self.project_id + "/agent"
        list_intents_req = gdf.types.ListIntentsRequest(
            parent=path,
            language_code=self.language)

        all_intents = client_intent.list_intents(parent=path)

        for intent in all_intents:
            if intent.display_name == "Default Fallback Intent" or intent.display_name == "Default Welcome Intent":
                delete_intent_request = gdf.types.DeleteIntentRequest(name=intent.name)
                try:
                    deleteIntent_response = client_intent.delete_intent(name=intent.name)
                except InvalidArgument:
                    raise
        return


    # startup method
    def dynamic_setup(self, df, name):
        intents = df["new_intent"].unique()
        all_examples = df[["utterance", "new_intent"]]
        if name == "mod_ir_all":
            name = "mod-ir-all"
        elif name == "mod_ir_self":
            name = "mod-ir-self" 
        self.deleteAgent()
        self.createAgent(name)
        self.createIntent(intents, all_examples)
        self.deleteDefaultIntents()
        return


    # send API requests for each utterance
    def chat(self, message):
        text_input = gdf.types.TextInput(text=message, language_code=self.language)
        query_input = gdf.types.QueryInput(text=text_input)

        try:
            response = self.session_client.detect_intent(session=self.session, query_input=query_input)
        except InvalidArgument:
            raise

        text = response.query_result.query_text
        intent = response.query_result.intent.display_name
        confidence = response.query_result.intent_detection_confidence
        # response_text = response.query_result.fulfillment_text
        response = IntentClassifierResponse(self.name, intent, confidence)
        
        return text, intent, confidence


# if __name__ == "__main__":

#     logging.basicConfig(level=logging.INFO)

#     dialogFlow = DialogFlowAPIInterface.createInstance()

#     if dialogFlow:
#         logging.info("create data for dialogflow")
#         data = pd.read_csv("data/dataset.csv")
#         dialogFlow_df = data[data["agent"] == "watson"]
#         dialogFlow.setup(dialogFlow_df)

#     dialogflow = DialogFlowAPIInterface.createInstance()
#     text, intent, confidence = dialogflow.chat("I don't care you anymore")
#     print("dialogflow returned intent \"{}\" with confidence {}".format(intent, confidence))
