import asyncio
from metagpt.actions import Action
from metagpt.roles import Role
from metagpt.logs import logger
import re

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

class SimpleCoder(Role):
    name: str = "Alice"
    profile:str = "SimpleCoder"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.__init__actions([SimpleWriteCode])
    async def _act(self) -> Message:
        logger.info(f"{self._setting}:ready to {self.rc.todo}")
        todo = self.rc.todo
        msg = self.get_memories(1)[0]
        code_result = await todo.run(msg.content)
        msg = Message(content=code_result, role=self.profile,cause_by=type(todo))
        return msg
        
async def main():
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)
asyncio.run(main())
