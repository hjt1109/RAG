from loguru import logger
import re 


class System_Recogni:
    """
    通过re提取 question中的&&包围的系统名称
    """
    def __init__(self):

        logger.info("System_Recogni init")

    def extract_system_name(self, question):
        """
        提取question中的系统名称
        """
        pattern = r"&&(.*?)&&"
        system_name = re.findall(pattern, question)
        if system_name:
            return system_name[0]
        else:
            return None



system_recogni = System_Recogni()
