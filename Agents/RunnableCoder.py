import asyncio
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.logs import logger
import re
import subprocess

from metagpt.schema import Message

class SimpleWriteCode(Action):
    PROMPT_TEMPLATE :str = """
    Write a python function that can {instruction} and provide two runnable test cases.
    Return ```python your_code_here``` with  NO other texts,
    your code:
    """
    name:str = "SimpleWriteCode"
    
    async def run(self, instruction:str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        rsp = await self._aask(prompt)
        code_text = SimpleWriteCode.parse_code(rsp)
        return code_text
    
    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp,re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text

class SimpleRunCode(Action):
    name:str = "SimpleRunCode"
    async def run(self, code_text):
        result = subprocess.run(["python3", "-c", code_text], capture_output=True,text=True)
        code_result = result.stdout
        logger.info(code_result)
        return code_result
    
class RunnableCoder(Role):
    name:str = "Alice"
    profile:str = "RunnableCoder"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.__init__actions([SimpleWriteCode,SimpleRunCode])
        self._set_react_mode(react_mode="by_order")
    async def _act(self) -> Message:
        logger.info(f"{self._setting}:ready to {self.rc.todo}")
        todo = self.rc.todo
        msg = self.get_memories(1)[0]
        result = await todo.run(msg.content)
        msg = Message(content=result, role=self.profile,cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg

async def main():
    msg = "write a function that calculates the sum of a list"
    role = RunnableCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)
asyncio.run(main())   
