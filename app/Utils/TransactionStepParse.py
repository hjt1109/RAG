import re
from loguru import logger


class TransactionStepParse:

    def transactionToSteps(self, question):
        """
        通用解析步骤，保留数字内容
        """
        try:
            pattern = r'(\d+)、(.*?)(?=\d+、|$)'
            matches = re.findall(pattern, question)

            steps = [f"{step_num}、{content.strip()}" for step_num, content in matches]
            return steps
        except Exception as e:
            logger.error(f"Error in transactionToSteps: {e}")
            return []

transactionStepParse = TransactionStepParse()