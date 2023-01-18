import configparser

class Config:

    config=None

    @staticmethod
    def init(configfile):
        Config.config = configparser.ConfigParser()
        Config.config.read(["testbed/default-config.ini", configfile])

    @staticmethod
    def get():
        if Config.config == None:
            raise Exception("Configuration is not initialised.")
        return Config.config
