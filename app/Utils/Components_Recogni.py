import re
from loguru import logger

class Components_Recogni:
    def __init__(self):

     
        logger.info("Components_Recogni Class Initiated")


    def get_components(self, question):
        """
        通过re提取 question中的< >中的组件名称，可能有多个，先提取成list在set去重复，再返回list
        """
        try:
            pattern = r'<(.*?)>'
            components = re.findall(pattern, question)
            components = list(set(components))
            components = [component.strip() for component in components]
            return components
        except Exception as e:
            self.logger.error("Error in get_components: {}".format(e))
            return []

components_recogni = Components_Recogni()
question = "What is the capital of <China> and <Russia>?  <China>  <1>"
components = components_recogni.get_components(question)
print(components)